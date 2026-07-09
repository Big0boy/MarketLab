import math
import random
import models
from models import OrderType
from order_book import OrderBook
from cda import CDA


class Exchange:

    def __init__(self, stocks: list[models.Stock],
                 slippage_factor: float = 0.0001,
                 max_short_pct: float = 0.10,
                 intraday_steps: int = 390,
                 holding_bonus_rate: float = 0.0002,
                 holding_bonus_interval: int = 30):
        self.order_book  = OrderBook()
        self.cda         = CDA(self.order_book, slippage_factor)
        self.max_short_pct      = max_short_pct
        self.intraday_steps     = intraday_steps
        self.holding_bonus_rate     = holding_bonus_rate      # bonus paid per share held long
        self.holding_bonus_interval = holding_bonus_interval  # steps between bonus payments
        self.agents      = {}
        self.trade_log   = []
        self.step        = 0
        self._last_earnings_events: list[models.EarningsEvent] = []  # cached for snapshot()

        for stock in stocks:
            self.order_book.stocks[stock.symbol] = stock

    def register_agent(self, agent) -> None:
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: int) -> None:
        """Remove agent AND all their outstanding orders"""
        if agent_id in self.agents:
            removed = self.order_book.remove_agent_orders(agent_id)
            # Change 5: wipe reservations — agent is gone
            agent = self.agents[agent_id]
            agent.reserved_cash = 0.0
            agent.reserved_holdings = {}
            del self.agents[agent_id]

    def _max_short(self, symbol: str) -> int:
        """Change 5: max short size is a percentage of the stock's float."""
        stock = self.order_book.stocks.get(symbol)
        if not stock:
            return 0
        return int(stock.total_shares * self.max_short_pct)

    def submit_order(self, order: models.Order) -> None:
        agent = self.agents.get(order.agent_id)
        if not agent:
            return
        if order.quantity <= 0 or order.price <= 0:
            return

        if order.order_type == OrderType.BUY:
            # Feature 3: snap limit price to tick grid
            order.price = self._snap_price(order.price, order.symbol)
            # Change 5: check available cash (total minus already reserved)
            available_cash = agent.cash - agent.reserved_cash
            cost = order.price * order.quantity
            if available_cash < cost:
                return

            # Change 5: a buy order can't require more shares than exist
            stock = self.order_book.stocks.get(order.symbol)
            if stock and order.quantity > stock.total_shares:
                return

            agent.reserved_cash += cost
            self.order_book.add_order(order)

        elif order.order_type == OrderType.SELL:
            # Feature 3: snap limit price to tick grid
            order.price = self._snap_price(order.price, order.symbol)
            # Change 5: check available holdings (owned minus already reserved)
            reserved_qty  = agent.reserved_holdings.get(order.symbol, 0)
            owned         = agent.holdings.get(order.symbol, 0)
            available_qty = owned - reserved_qty
            max_short     = self._max_short(order.symbol)
            if available_qty - order.quantity >= -max_short:
                agent.reserved_holdings[order.symbol] = reserved_qty + order.quantity
                self.order_book.add_order(order)

    def run_step(self) -> tuple[list[models.Trade], list[models.EarningsEvent]]:
        # Earnings announcements: fire before auction so agents react same step
        earnings_events = self._fire_earnings()
        self._last_earnings_events = earnings_events  # cache for snapshot()

        # Change 1: expire stale orders and release their reservations
        expired = self.order_book.expire_orders(self.step)
        for order in expired:
            self._release_reservation(order)

        # Holding bonus: reward long holders every holding_bonus_interval steps
        if self.holding_bonus_interval > 0 and self.step % self.holding_bonus_interval == 0:
            self._pay_holding_bonus()

        # Change 1: charge borrow fees on short positions before the auction
        self._charge_borrow_fees()

        # Change 3: only auction stocks that aren't halted
        # Feature 5: scale matched volume by intraday U-curve
        vol_mult = self._intraday_volume_multiplier()
        trades = self.cda.run_auction(self.step,
                                      skip_symbols=self._halted_symbols(),
                                      volume_multiplier=vol_mult)

        for trade in trades:
            self._settle_trade(trade)

        self._apply_shock()
        self._check_margin_calls()

        for symbol in list(self.order_book.stocks.keys()):
            self._check_circuit_breaker(symbol)

        for stock in self.order_book.stocks.values():
            stock.price_history.append(round(stock.price, 4))
            # Change 3: count down the halt timer
            if stock.halted_steps > 0:
                stock.halted_steps -= 1

        self.step += 1
        return trades, earnings_events

    def _fire_earnings(self) -> list[models.EarningsEvent]:
        """Fire earnings announcements for any stock due this step.

        True fundamental jumps immediately. A noisy public signal is
        broadcast so agents with heterogeneous beliefs can update at
        different rates. A dividend is paid to all long holders —
        agents that hold through earnings receive a concrete cash reward,
        giving them a rational incentive to be patient rather than churning.
        Short sellers pay the dividend (realistic: short sellers owe
        dividends to the lender).
        """
        events = []
        for stock in self.order_book.stocks.values():
            if self.step < stock.next_earnings:
                continue

            # True earnings surprise — drives actual fundamental shift
            true_surprise = random.gauss(0, stock.earnings_surprise_vol)
            stock.fundamental_value *= (1 + true_surprise)
            stock.fundamental_value  = max(stock.fundamental_value, 0.01)

            # Public signal: agents observe this, not the true surprise
            signal_noise  = random.gauss(0, stock.earnings_surprise_vol * 0.5)
            public_signal = true_surprise + signal_noise

            # Price reacts immediately to signal (partial, 30% of signal)
            stock.price *= (1 + 0.3 * public_signal)
            stock.price  = max(stock.price, 0.01)

            # Earnings day spikes turbulence
            stock._turb = 30  # type: ignore[attr-defined]

            # Pay dividend: dividend_per_share = price * dividend_yield
            # Long holders receive it; short sellers pay it (realistic)
            dividend_per_share = stock.price * stock.dividend_yield
            for agent in self.agents.values():
                qty = agent.holdings.get(stock.symbol, 0)
                if qty != 0:
                    # positive qty → receive; negative qty → pay
                    agent.cash += qty * dividend_per_share

            event = models.EarningsEvent(
                symbol             = stock.symbol,
                step               = self.step,
                true_surprise      = true_surprise,
                public_signal      = public_signal,
                dividend_per_share = dividend_per_share,
            )
            stock.earnings_history.append(event)
            stock.next_earnings += stock.earnings_interval
            events.append(event)

        return events

    def _halted_symbols(self) -> set[str]:
        """Change 3: symbols currently under a circuit-breaker halt."""
        return {
            symbol for symbol, stock in self.order_book.stocks.items()
            if stock.halted_steps > 0
        }

    def _intraday_volume_multiplier(self) -> float:
        """Feature 5: U-shaped intraday volume curve.
        Peaks at open (step 0) and close (step intraday_steps-1),
        troughs at midday. Modelled as a cosine so it's smooth and
        symmetric. Returns a multiplier in [0.5, 2.0].
        """
        t = (self.step % self.intraday_steps) / max(self.intraday_steps - 1, 1)
        # cos(2πt) peaks at 0 and 1 (open/close), troughs at 0.5 (midday)
        raw   = math.cos(2 * math.pi * t)          # [-1, 1]
        scale = 0.5 + 0.75 * (raw + 1)             # [0.5, 2.0]
        return scale

    def _snap_price(self, price: float, symbol: str) -> float:
        """Feature 3: round price to nearest tick for the given stock."""
        stock = self.order_book.stocks.get(symbol)
        if not stock or stock.tick_size <= 0:
            return price
        return round(round(price / stock.tick_size) * stock.tick_size, 10)

    def _charge_borrow_fees(self) -> None:
        """Change 1: charge each agent a per-step fee for shares held short."""
        for agent in self.agents.values():
            for symbol, qty in agent.holdings.items():
                if qty < 0:
                    stock = self.order_book.stocks.get(symbol)
                    if stock:
                        fee = abs(qty) * stock.price * stock.borrow_rate
                        agent.cash -= fee

    def _pay_holding_bonus(self) -> None:
        """Pay a small cash bonus to agents holding long positions,
        rewarding patient capital and discouraging pure churn."""
        for agent in self.agents.values():
            for symbol, qty in agent.holdings.items():
                if qty > 0:
                    stock = self.order_book.stocks.get(symbol)
                    if stock:
                        bonus = qty * stock.price * self.holding_bonus_rate
                        agent.cash += bonus

    def _release_reservation(self, order: models.Order) -> None:
        """Change 5: free reserved balance when an order is cancelled/expired."""
        agent = self.agents.get(order.agent_id)
        if not agent:
            return
        if order.order_type == OrderType.BUY:
            cost = order.price * order.quantity
            agent.reserved_cash = max(0.0, agent.reserved_cash - cost)
        else:
            agent.reserved_holdings[order.symbol] = max(
                0, agent.reserved_holdings.get(order.symbol, 0) - order.quantity
            )

    def _settle_trade(self, trade: models.Trade) -> None:
        buyer  = self.agents.get(trade.buyer_id)
        seller = self.agents.get(trade.seller_id)

        if not buyer or not seller:
            return

        total_cost = trade.price * trade.quantity
        fee        = 0.00025 * total_cost   # halved from 0.0005

        buyer.cash                   -= total_cost + fee
        buyer.holdings[trade.symbol]  = buyer.holdings.get(trade.symbol, 0) + trade.quantity
        # Change 5: release the reservation for the filled portion only.
        # The CDA mutates order.quantity in-place for partial fills, so the
        # remaining portion stays reserved via the live order still on the book.
        buyer.reserved_cash = max(0.0, buyer.reserved_cash - total_cost)

        seller.cash                  += total_cost - fee
        seller.holdings[trade.symbol] = seller.holdings.get(trade.symbol, 0) - trade.quantity
        # Change 5: release the holding reservation for the seller
        seller.reserved_holdings[trade.symbol] = max(
            0, seller.reserved_holdings.get(trade.symbol, 0) - trade.quantity
        )

        self.trade_log.append(trade)

    def _check_margin_calls(self) -> None:
        """Liquidate bankrupt agents and remove them completely"""
        for agent_id, agent in list(self.agents.items()):
            if agent.cash < 0:
                # liquidate short positions
                for symbol, qty in list(agent.holdings.items()):
                    if qty < 0:
                        stock = self.order_book.stocks.get(symbol)
                        if stock:
                            agent.cash += qty * stock.price  # qty is negative
                            agent.holdings[symbol] = 0

                # ❗ If still bankrupt → remove agent entirely
                if agent.cash < 0:
                    self.remove_agent(agent_id)

    def _apply_shock(self) -> None:
        # Macro shock: single draw shared across all stocks this step.
        # Models market-wide sentiment shifts (Fed, macro news, risk-off).
        # Stocks with higher volatility have larger macro loading.
        macro_shock = random.gauss(0, 0.001)

        for stock in self.order_book.stocks.values():

            shock     = random.gauss(0, 0.002)
            long_term = stock.initial_fundamental
            # Strengthened mean reversion (0.003 vs old 0.001)
            reversion = 0.003 * (long_term - stock.fundamental_value)

            stock.fundamental_value *= (1 + shock + reversion)
            stock.fundamental_value  = max(stock.fundamental_value, 0.01)

            # Slowed sediment decay (0.98 vs old 0.95) — sentiment persists longer
            stock.sediment = 0.98 * stock.sediment + shock * 0.1

            if not hasattr(stock, '_turb'):
                stock._turb = 0
            if stock._turb > 0:
                stock._turb -= 1

            vol_scale = 0.15 + (stock._turb / 50) * 0.45

            # Apply idiosyncratic shock + macro shock (scaled by volatility)
            idio_shock  = random.gauss(0, stock.volatility * vol_scale)
            macro_load  = macro_shock * (stock.volatility / 0.02)  # normalised to 0.02 base
            stock.price *= (1 + idio_shock + macro_load)

            gap = (stock.fundamental_value - stock.price) / stock.price
            stock.price *= (1 + 0.01 * gap)
            stock.price  = max(stock.price, 0.01)

            if random.random() < 0.02:
                news         = random.gauss(0, stock.volatility * 0.5)
                stock.price *= (1 + news)
                stock._turb  = 50

    def _check_circuit_breaker(self, symbol: str) -> None:
        stock = self.order_book.stocks.get(symbol)
        if not stock or not stock.price_history:
            return

        last_price = stock.price_history[-1]
        if last_price == 0:
            return

        change = abs(stock.price - last_price) / last_price
        if change > 0.10:
            self.order_book.bids.pop(symbol, None)
            self.order_book.asks.pop(symbol, None)
            # Change 3: sentiment decay + trading halt on circuit breaker
            stock.sediment     = 0.0
            stock.halted_steps = 5

    def get_price(self, symbol: str) -> float | None:
        stock = self.order_book.stocks.get(symbol)
        # Change 3: halted stocks return None so agents skip their turn
        if stock and stock.halted_steps > 0:
            return None

        mid = self.cda.calculate_price(symbol)
        if mid:
            return mid

        return stock.price if stock else None

    def get_fundamental_value(self, symbol: str) -> float | None:
        stock = self.order_book.stocks.get(symbol)
        return stock.fundamental_value if stock else None

    def get_price_history(self, symbol: str) -> list[float] | None:
        stock = self.order_book.stocks.get(symbol)
        return stock.price_history if stock else None

    def get_spread(self, symbol: str) -> float | None:
        return self.order_book.get_spread(symbol)

    def get_trade_log(self) -> list[models.Trade]:
        return self.trade_log

    def snapshot(self, earnings_events: list[models.EarningsEvent] | None = None) -> "MarketSnapshot":
        # Default to the most recently fired events so callers don't need to
        # pass them explicitly — fixes the design gap where snapshot() always
        # returned an empty earnings_events list.
        if earnings_events is None:
            earnings_events = self._last_earnings_events
        prices           = {}
        fundamentals     = {}
        histories        = {}
        spreads          = {}
        sediments        = {}
        bid_depth        = {}
        ask_depth        = {}
        max_shorts       = {}
        advs             = {}
        pending_earnings = {}
        dividend_yields  = {}   # symbol -> dividend_yield (agents use this to value hold)

        for symbol, stock in self.order_book.stocks.items():
            mid = self.cda.calculate_price(symbol)
            prices[symbol]           = mid if mid else stock.price
            fundamentals[symbol]     = stock.fundamental_value
            histories[symbol]        = list(stock.price_history)
            sediments[symbol]        = stock.sediment
            max_shorts[symbol]       = self._max_short(symbol)
            advs[symbol]             = stock.adv
            pending_earnings[symbol] = max(0, stock.next_earnings - self.step)
            dividend_yields[symbol]  = stock.dividend_yield

            bid = self.order_book.get_best_bid(symbol)
            ask = self.order_book.get_best_ask(symbol)
            spreads[symbol] = (round(ask.price - bid.price, 4)
                               if bid and ask else None)

            bid_depth[symbol] = sum(
                o.quantity
                for level in self.order_book.bids.get(symbol, {}).values()
                for o in level
            )
            ask_depth[symbol] = sum(
                o.quantity
                for level in self.order_book.asks.get(symbol, {}).values()
                for o in level
            )

        return MarketSnapshot(
            step              = self.step,
            prices            = prices,
            fundamentals      = fundamentals,
            histories         = histories,
            sediments         = sediments,
            spreads           = spreads,
            bid_depth         = bid_depth,
            ask_depth         = ask_depth,
            max_short         = max_shorts,
            halted            = self._halted_symbols(),
            volume_multiplier = self._intraday_volume_multiplier(),
            advs              = advs,
            earnings_events   = earnings_events or [],
            pending_earnings  = pending_earnings,
            dividend_yields   = dividend_yields,
        )


class MarketSnapshot:
    __slots__ = (
        "step", "prices", "fundamentals", "histories",
        "sediments", "spreads", "bid_depth", "ask_depth", "max_short",
        "halted", "volume_multiplier", "advs",
        "earnings_events", "pending_earnings", "dividend_yields",
    )

    def __init__(self, step, prices, fundamentals, histories,
                 sediments, spreads, bid_depth, ask_depth, max_short,
                 halted=None, volume_multiplier=1.0, advs=None,
                 earnings_events=None, pending_earnings=None,
                 dividend_yields=None):
        self.step              = step
        self.prices            = prices
        self.fundamentals      = fundamentals
        self.histories         = histories
        self.sediments         = sediments
        self.spreads           = spreads
        self.bid_depth         = bid_depth
        self.ask_depth         = ask_depth
        self.max_short         = max_short
        self.halted            = halted or set()
        self.volume_multiplier = volume_multiplier
        self.advs              = advs or {}
        self.earnings_events   = earnings_events or []
        self.pending_earnings  = pending_earnings or {}
        self.dividend_yields   = dividend_yields or {}

    # ── Convenience accessors that mirror Exchange's public API ──────────────

    def get_price(self, symbol: str) -> float | None:
        if symbol in self.halted:
            return None
        return self.prices.get(symbol)

    def get_fundamental_value(self, symbol: str) -> float | None:
        return self.fundamentals.get(symbol)

    def get_price_history(self, symbol: str) -> list[float] | None:
        return self.histories.get(symbol)

    def get_spread(self, symbol: str) -> float | None:
        return self.spreads.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.keys())