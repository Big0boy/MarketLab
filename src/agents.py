from __future__ import annotations
import math
import random
import statistics
from typing import TYPE_CHECKING
from models import Order, OrderType, AgentType, MIN_ORDER_SIZE

if TYPE_CHECKING:
    from exchange import Exchange


class BaseAgent:

    agent_type: AgentType = AgentType.Base

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        self.agent_id          = agent_id
        self.cash              = initial_cash
        self.holdings          = {}
        self.trade_history     = []
        self.wealth_history    = []
        self.raw_fitness       = 0
        self.fitness           = 0
        # Change 5: reserved balance tracking (mirrors Exchange-side)
        self.reserved_cash     = 0.0
        self.reserved_holdings = {}
        # Change 4: above this portfolio_value/total_wealth ratio, dampen buys
        self.cash_preference   = cash_preference
        # Change 6: cap on holdings as a fraction of a stock's total float
        self.max_ownership_pct = max_ownership_pct

    def decide(self, exchange: Exchange) -> list[Order]:
        # Change 3: all agents return a list (empty = no action)
        return []

    def get_total_wealth(self, exchange: Exchange) -> float:
        total = self.cash
        for symbol, quantity in self.holdings.items():
            stock = exchange.order_book.stocks.get(symbol)
            if stock:
                total += quantity * stock.price
        return total

    def portfolio_value(self, exchange: Exchange) -> float:
        return sum(
            qty * exchange.order_book.stocks[sym].price
            for sym, qty in self.holdings.items()
            if sym in exchange.order_book.stocks
        )

    def record_wealth(self, exchange: Exchange) -> None:
        self.wealth_history.append(self.get_total_wealth(exchange))

    def is_bankrupt(self, exchange: Exchange, threshold: float = 100.0) -> bool:
        return self.get_total_wealth(exchange) < threshold

    def calculate_order_size(self,
                             conviction: float,
                             price: float,
                             risk_aversion: float,
                             exchange: Exchange,
                             symbol: str | None = None,
                             order_type: OrderType = OrderType.BUY) -> int:
        wealth   = self.get_total_wealth(exchange)
        raw_size = (conviction * wealth) / (price * risk_aversion)

        # Change 4: cash-preference dampening on buys when already
        # heavily invested
        if order_type == OrderType.BUY and wealth > 0:
            invested_ratio = self.portfolio_value(exchange) / wealth
            if invested_ratio > self.cash_preference:
                excess     = invested_ratio - self.cash_preference
                dampening  = max(0.0, 1 - excess / (1 - self.cash_preference))
                raw_size  *= dampening

        # Feature 4: scale by depth imbalance when data is available
        # Agents are more aggressive when book is deep on their side
        if symbol is not None:
            bid_d = sum(
                o.quantity
                for level in exchange.order_book.bids.get(symbol, {}).values()
                for o in level
            )
            ask_d = sum(
                o.quantity
                for level in exchange.order_book.asks.get(symbol, {}).values()
                for o in level
            )
            total_depth = bid_d + ask_d
            if total_depth > 0:
                if order_type == OrderType.BUY:
                    # more bids = more liquidity on our side → scale up slightly
                    depth_factor = 0.8 + 0.4 * (bid_d / total_depth)
                else:
                    depth_factor = 0.8 + 0.4 * (ask_d / total_depth)
                raw_size *= depth_factor

        size = max(MIN_ORDER_SIZE, int(raw_size))

        # Change 6 (revised): cap a single order at a fraction of wealth/price
        # rather than a flat share count, so cheap and expensive stocks aren't
        # squashed to the same distribution.
        max_value_per_order = 0.15 * wealth
        max_size_by_value   = max(MIN_ORDER_SIZE, int(max_value_per_order / price))
        size = min(size, max_size_by_value)

        # Change 6: cap resulting holdings at max_ownership_pct of float
        if symbol is not None:
            stock = exchange.order_book.stocks.get(symbol)
            if stock and stock.total_shares > 0:
                max_holding = int(stock.total_shares * self.max_ownership_pct)
                current     = self.holdings.get(symbol, 0)

                if order_type == OrderType.BUY:
                    room = max_holding - current
                else:  # SELL — limit how short the agent can go
                    room = max_holding + current

                room = max(0, room)
                size = min(size, room)

        return max(0, size)

    def __repr__(self) -> str:
        return (f"{self.agent_type.value}("
                f"id={self.agent_id}, "
                f"cash={self.cash:.2f}, "
                f"holdings={self.holdings})")

# FUNDAMENTALIST
class FundamentalistAgent(BaseAgent):
    agent_type = AgentType.FUNDAMENTALIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 buy_threshold:    float = 0.98,
                 sell_threshold:   float = 1.02,
                 risk_aversion:    float = 1.0,
                 cash_preference:  float = 0.8,
                 max_ownership_pct: float = 0.30,
                 # Heterogeneous beliefs parameters
                 belief_noise:     float = 0.05,   # σ of initial private estimate error
                 learning_rate:    float = 0.3,    # how fast to update on earnings signal
                 signal_sensitivity: float = 1.0): # scales how much of the signal is absorbed
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.buy_threshold      = buy_threshold
        self.sell_threshold     = sell_threshold
        self.risk_aversion      = risk_aversion
        self.belief_noise       = belief_noise
        self.learning_rate      = learning_rate
        self.signal_sensitivity = signal_sensitivity
        # Private fundamental estimates: initialised lazily on first decide()
        # so we have access to actual stock fundamentals at runtime
        self._private_estimates: dict[str, float] = {}
        # Confidence per symbol: starts low, grows as agent sees more earnings
        self._confidence: dict[str, float] = {}
        # Track which earnings events we've already processed
        self._last_earnings_step: dict[str, int] = {}

    def _get_private_estimate(self, symbol: str,
                               exchange: Exchange) -> float:
        """Return (and lazily initialise) this agent's private fair-value estimate."""
        if symbol not in self._private_estimates:
            true_fv = exchange.get_fundamental_value(symbol) or 1.0
            # Each agent starts with a noisy version of the true fundamental
            noise   = random.gauss(0, self.belief_noise * true_fv)
            self._private_estimates[symbol] = max(true_fv + noise, 0.01)
            self._confidence[symbol]        = 0.3   # low initial confidence
        return self._private_estimates[symbol]

    def update_beliefs(self, exchange: Exchange) -> None:
        """Call once per step to incorporate any earnings announcements.

        Each agent observes the public signal but updates by a different
        amount based on their learning_rate and current confidence.
        High-confidence agents trust their own model more and update less;
        low-confidence agents are more market-driven.
        """
        for stock in exchange.order_book.stocks.values():
            symbol = stock.symbol
            if not stock.earnings_history:
                continue
            latest = stock.earnings_history[-1]
            # Only process each event once
            if self._last_earnings_step.get(symbol, -1) == latest.step:
                continue
            self._last_earnings_step[symbol] = latest.step

            # Agent observes public signal with additional private noise
            # Higher confidence → less private noise → closer to true signal
            conf         = self._confidence.get(symbol, 0.3)
            private_noise = random.gauss(
                0, stock.earnings_surprise_vol * (1 - conf * 0.8)
            )
            perceived_signal = (latest.public_signal + private_noise) * self.signal_sensitivity

            # Bayesian-ish update: blend old estimate toward signal-implied value
            old_estimate = self._get_private_estimate(symbol, exchange)
            # Signal implies fundamental moved by perceived_signal fraction
            signal_implied = old_estimate * (1 + perceived_signal)
            new_estimate   = (old_estimate * (1 - self.learning_rate)
                              + signal_implied * self.learning_rate)
            self._private_estimates[symbol] = max(new_estimate, 0.01)

            # Confidence grows with each earnings event seen (up to 0.9)
            self._confidence[symbol] = min(conf + 0.05, 0.90)

    def decide(self, exchange: Exchange) -> list[Order]:
        # Update private beliefs from any earnings this step
        self.update_beliefs(exchange)

        best_order     = None
        best_deviation = 0.0

        for symbol in exchange.order_book.stocks:
            price    = exchange.get_price(symbol)
            if price is None:
                continue

            # Use private estimate instead of shared fundamental_value
            estimate = self._get_private_estimate(symbol, exchange)
            deviation  = abs(price - estimate) / estimate
            sediment   = exchange.order_book.stocks[symbol].sediment

            # Boost conviction when earnings are imminent (pre-earnings positioning)
            stock = exchange.order_book.stocks[symbol]
            steps_to_earnings = max(stock.next_earnings - exchange.step, 1)
            earnings_proximity = 1.0 + max(0, (10 - steps_to_earnings) / 10) * 0.5

            # Dividend-aware: if dividend is imminent, boost buy / dampen sell.
            # Agents know dividend_yield and steps_to_earnings from the stock,
            # so they can rationally value the upcoming cash flow.
            # Expected dividend value as fraction of price = yield * proximity
            div_proximity = max(0, (20 - steps_to_earnings) / 20)
            div_value_frac = stock.dividend_yield * div_proximity
            # Boost buy conviction by expected dividend value
            buy_boost  = 1.0 + div_value_frac * 5
            # Dampen sell conviction — selling before dividend means missing payout
            sell_dampen = max(0.3, 1.0 - div_value_frac * 4)

            buy_conviction  = min(
                deviation * 2 * (1 + max(sediment, 0)) * earnings_proximity * buy_boost,
                1.0
            )
            sell_conviction = min(
                deviation * 2 * (1 + max(-sediment, 0)) * earnings_proximity * sell_dampen,
                1.0
            )

            if deviation > best_deviation:
                if price < estimate * self.buy_threshold:
                    quantity = self.calculate_order_size(buy_conviction, price,
                                                         self.risk_aversion, exchange,
                                                         symbol=symbol, order_type=OrderType.BUY)
                    if buy_conviction > 0.6:
                        best_ask  = exchange.order_book.get_best_ask(symbol)
                        bid_price = (best_ask.price * 1.001
                                     if best_ask else price * 1.005)
                    else:
                        bid_price = price * 1.005
                    best_order     = Order(self.agent_id, symbol, OrderType.BUY,
                                          bid_price, quantity, exchange.step, ttl=20)
                    best_deviation = deviation

                elif price > estimate * self.sell_threshold:
                    quantity = self.calculate_order_size(sell_conviction, price,
                                                         self.risk_aversion, exchange,
                                                         symbol=symbol, order_type=OrderType.SELL)
                    if sell_conviction > 0.6:
                        best_bid  = exchange.order_book.get_best_bid(symbol)
                        ask_price = (best_bid.price * 0.999
                                     if best_bid else price * 0.995)
                    else:
                        ask_price = price * 0.995
                    best_order     = Order(self.agent_id, symbol, OrderType.SELL,
                                          ask_price, quantity, exchange.step, ttl=20)
                    best_deviation = deviation

        return [best_order] if best_order else []

#CHARTIST
class ChartistAgent(BaseAgent):
    agent_type = AgentType.CHARTIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 5,
                 risk_aversion: float = 1.2,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.lookback      = lookback
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> list[Order]:
        best_order = None
        best_trend = 0.0

        for symbol in exchange.order_book.stocks:
            history = exchange.get_price_history(symbol)
            if not history or len(history) < self.lookback:
                continue

            # Build set of list indices (not step numbers) to exclude.
            # price_history grows by one entry per step, so index = step number
            # when history started at step 0 — but we map via step directly to
            # avoid any off-by-one if history was pre-seeded.
            stock = exchange.order_book.stocks[symbol]
            earnings_indices: set[int] = {ev.step for ev in stock.earnings_history}

            # Filter out earnings-contaminated indices from lookback window
            clean_history = [
                p for i, p in enumerate(history)
                if i not in earnings_indices
            ]
            if len(clean_history) < self.lookback:
                continue

            price_now  = clean_history[-1]
            price_then = clean_history[-self.lookback]
            if price_then == 0:
                continue

            sediment   = exchange.order_book.stocks[symbol].sediment
            raw_trend  = (price_now - price_then) / price_then
            aligned    = (raw_trend >= 0 and sediment >= 0) or (raw_trend < 0 and sediment < 0)
            alignment  = (1 + sediment) if aligned else (1 - abs(sediment) * 0.5)
            trend      = raw_trend * max(alignment, 0.1)
            conviction = min(abs(trend) * 5, 1.0)
            order_dir  = OrderType.BUY if trend > 0 else OrderType.SELL
            quantity   = self.calculate_order_size(conviction, price_now,
                                                   self.risk_aversion, exchange,
                                                   symbol=symbol, order_type=order_dir)

            if abs(trend) > abs(best_trend):
                if trend > 0:
                    # Change 2: strong trend → cross the spread
                    if conviction > 0.5:
                        best_ask  = exchange.order_book.get_best_ask(symbol)
                        bid_price = best_ask.price * 1.001 if best_ask else price_now * 1.005
                    else:
                        bid_price = price_now * 1.005
                    best_order = Order(self.agent_id, symbol, OrderType.BUY,
                                       bid_price, quantity, exchange.step, ttl=8)
                    best_trend = trend
                elif trend < 0:
                    if conviction > 0.5:
                        best_bid  = exchange.order_book.get_best_bid(symbol)
                        ask_price = best_bid.price * 0.999 if best_bid else price_now * 0.995
                    else:
                        ask_price = price_now * 0.995
                    best_order = Order(self.agent_id, symbol, OrderType.SELL,
                                       ask_price, quantity, exchange.step, ttl=8)
                    best_trend = trend

        return [best_order] if best_order else []

# NOISE TRADER
class NoiseAgent(BaseAgent):
    agent_type = AgentType.NOISE

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 hold_prob:float = 0.40,
                 buy_prob:float = 0.30,
                 risk_aversion: float = 2.0,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.hold_prob     = hold_prob
        self.buy_prob      = buy_prob
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> list[Order]:
        # Feature 5: noise traders are more active at open/close
        t = (exchange.step % max(exchange.intraday_steps, 1)) / max(exchange.intraday_steps - 1, 1)
        vol_mult    = 0.5 + 0.75 * (math.cos(2 * math.pi * t) + 1)
        active_prob = max(0.0, min(1.0 - self.hold_prob * (1.5 / vol_mult), 0.95))
        if random.random() > active_prob:
            return []

        symbols = list(exchange.order_book.stocks.keys())
        if not symbols:
            return []

        symbol = random.choice(symbols)
        price  = exchange.get_price(symbol)
        if price is None:
            return []

        sediment   = exchange.order_book.stocks[symbol].sediment
        conviction = random.uniform(0.1, 0.4) * (1 + abs(sediment))
        roll = random.random()
        buy_threshold = self.buy_prob * (1 + sediment * 0.5)
        order_dir = OrderType.BUY if roll < buy_threshold else OrderType.SELL
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange,
                                               symbol=symbol, order_type=order_dir)

        # Change 2: noise traders occasionally cross the spread (market orders)
        cross_spread = random.random() < 0.3

        if roll < buy_threshold:
            if cross_spread:
                best_ask  = exchange.order_book.get_best_ask(symbol)
                bid_price = best_ask.price * 1.001 if best_ask else price * 1.01
            else:
                bid_price = price * 1.005
            return [Order(self.agent_id, symbol, OrderType.BUY,
                          bid_price, quantity, exchange.step, ttl=2)]
        else:
            if cross_spread:
                best_bid  = exchange.order_book.get_best_bid(symbol)
                ask_price = best_bid.price * 0.999 if best_bid else price * 0.99
            else:
                ask_price = price * 0.995
            return [Order(self.agent_id, symbol, OrderType.SELL,
                          ask_price, quantity, exchange.step, ttl=2)]

# MARKET MAKER
class MarketMakerAgent(BaseAgent):
    agent_type = AgentType.MARKET_MAKER

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 spread_pct:    float = 0.004,
                 risk_aversion: float = 0.5,
                 target_inventory: int = 0,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.spread_pct      = spread_pct
        self.risk_aversion   = risk_aversion
        # Change 2: desired neutral inventory level per stock
        self.target_inventory = target_inventory

    def decide(self, exchange: Exchange) -> list[Order]:
        # Change 3: market maker now returns all its quotes as a list
        # so simulation._step() handles submission uniformly
        symbols = list(exchange.order_book.stocks.keys())
        n       = len(symbols) or 1
        orders  = []

        for symbol in symbols:
            price = exchange.get_price(symbol)
            if price is None:
                continue

            sediment   = exchange.order_book.stocks[symbol].sediment
            conviction = 0.3 * (1 + abs(sediment))
            quantity   = self.calculate_order_size(conviction, price,
                                                   self.risk_aversion, exchange,
                                                   symbol=symbol, order_type=OrderType.BUY)

            # Change 2: inventory-aware skew. Long → skew quotes down to
            # offload inventory; short → skew quotes up to cover.
            stock          = exchange.order_book.stocks[symbol]
            net_position   = self.holdings.get(symbol, 0) - self.target_inventory
            target_range   = max(1, int(stock.total_shares * self.max_ownership_pct))
            inventory_frac = max(-1.0, min(net_position / target_range, 1.0))
            skew           = -inventory_frac * (self.spread_pct / 2)

            bid_price = round(price * (1 - self.spread_pct / 2 + skew), 4)
            ask_price = round(price * (1 + self.spread_pct / 2 + skew), 4)

            # Cap both sides symmetrically: spread cash evenly across all symbols
            affordable = max(MIN_ORDER_SIZE, int(self.cash / (bid_price * n)))
            bid_qty    = min(quantity, affordable)
            ask_qty    = min(
                self.calculate_order_size(conviction, price,
                                          self.risk_aversion, exchange,
                                          symbol=symbol, order_type=OrderType.SELL),
                affordable,   # same per-symbol cap as buy side
            )

            # Change 1: short TTL so quotes stay fresh
            orders.append(Order(self.agent_id, symbol, OrderType.BUY,
                                bid_price, bid_qty, exchange.step, ttl=3))
            orders.append(Order(self.agent_id, symbol, OrderType.SELL,
                                ask_price, ask_qty, exchange.step, ttl=3))

        return orders

class MomentumAgent(BaseAgent):
    agent_type = AgentType.MOMENTUM

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 10,
                 threshold:     float = 0.005,
                 risk_aversion: float = 1.5,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.lookback      = lookback       
        self.threshold     = threshold      
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> list[Order]:
        best_symbol    = None
        best_signal    = 0.0

        for symbol in exchange.order_book.stocks:
            history = exchange.get_price_history(symbol)
            if not history or len(history) < self.lookback:
                continue

            # Build set of list indices for this symbol's earnings events.
            stock = exchange.order_book.stocks[symbol]
            earnings_indices: set[int] = {ev.step for ev in stock.earnings_history}

            # Clean history: remove earnings-day spikes
            clean_history = [
                p for i, p in enumerate(history)
                if i not in earnings_indices
            ]
            if len(clean_history) < self.lookback:
                continue

            price_now  = clean_history[-1]
            price_then = clean_history[-self.lookback]
            if price_then == 0:
                continue

            momentum = (price_now - price_then) / price_then
            if abs(momentum) > abs(best_signal) and abs(momentum) > self.threshold:
                best_signal = momentum
                best_symbol = symbol

        if best_symbol is None:
            return []

        price = exchange.get_price(best_symbol)
        if price is None:
            return []

        conviction = min(abs(best_signal) * 10, 1.0)
        # Feature 5: momentum agents are more aggressive at high-volume periods
        t = (exchange.step % max(exchange.intraday_steps, 1)) / max(exchange.intraday_steps - 1, 1)
        vol_mult   = 0.5 + 0.75 * (math.cos(2 * math.pi * t) + 1)
        conviction = min(conviction * (0.7 + 0.3 * vol_mult), 1.0)
        order_dir  = OrderType.BUY if best_signal > 0 else OrderType.SELL
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange,
                                               symbol=best_symbol, order_type=order_dir)

        # Change 2: momentum agents ride the trend aggressively — always cross
        if best_signal > 0:
            best_ask  = exchange.order_book.get_best_ask(best_symbol)
            bid_price = best_ask.price * 1.001 if best_ask else price * 1.005
            return [Order(self.agent_id, best_symbol, OrderType.BUY,
                          bid_price, quantity, exchange.step, ttl=5)]
        else:
            best_bid  = exchange.order_book.get_best_bid(best_symbol)
            ask_price = best_bid.price * 0.999 if best_bid else price * 0.995
            return [Order(self.agent_id, best_symbol, OrderType.SELL,
                          ask_price, quantity, exchange.step, ttl=5)]

class MeanReversionAgent(BaseAgent):
    agent_type = AgentType.MEAN_REVERSION

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 20,
                 entry_z:       float = 0.01,
                 risk_aversion: float = 1.2,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.lookback      = lookback    # rolling mean window
        self.entry_z       = entry_z     # minimum deviation fraction to act
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> list[Order]:
        best_symbol    = None
        best_deviation = 0.0
        best_direction = None

        for symbol in exchange.order_book.stocks:
            history = exchange.get_price_history(symbol)
            if not history or len(history) < self.lookback:
                continue

            window    = history[-self.lookback:]
            mean      = sum(window) / len(window)
            price_now = history[-1]
            if mean == 0:
                continue

            deviation = (price_now - mean) / mean

            if abs(deviation) > abs(best_deviation) and abs(deviation) > self.entry_z:
                best_deviation = deviation
                best_symbol    = symbol
                best_direction = OrderType.SELL if deviation > 0 else OrderType.BUY

        if best_symbol is None:
            return []

        price = exchange.get_price(best_symbol)
        if price is None:
            return []

        conviction = min(abs(best_deviation) * 20, 1.0)
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange,
                                               symbol=best_symbol, order_type=best_direction)

        if best_direction == OrderType.BUY:
            return [Order(self.agent_id, best_symbol, OrderType.BUY,
                          price * 1.005, quantity, exchange.step, ttl=15)]
        else:
            return [Order(self.agent_id, best_symbol, OrderType.SELL,
                          price * 0.995, quantity, exchange.step, ttl=15)]

class HerdAgent(BaseAgent):
    agent_type = AgentType.HERD

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 sensitivity:   float = 1.0,
                 risk_aversion: float = 1.8,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.sensitivity   = sensitivity  
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> list[Order]:
        best_symbol    = None
        best_imbalance = 0.0

        for symbol in exchange.order_book.stocks:
            bids = exchange.order_book.bids.get(symbol, {})
            asks = exchange.order_book.asks.get(symbol, {})

            total_bid_qty = sum(
                o.quantity for level in bids.values() for o in level
            )
            total_ask_qty = sum(
                o.quantity for level in asks.values() for o in level
            )

            total = total_bid_qty + total_ask_qty
            if total == 0:
                continue

            imbalance = (total_bid_qty - total_ask_qty) / total
            sediment  = exchange.order_book.stocks[symbol].sediment
            signal    = imbalance * (1 + abs(sediment) * self.sensitivity)
            signal    = max(-1.0, min(signal, 1.0))

            if abs(signal) > abs(best_imbalance):
                best_imbalance = signal
                best_symbol    = symbol

        if best_symbol is None or abs(best_imbalance) < 0.05:
            return []

        price = exchange.get_price(best_symbol)
        if price is None:
            return []

        conviction = min(abs(best_imbalance), 1.0)
        order_dir  = OrderType.BUY if best_imbalance > 0 else OrderType.SELL
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange,
                                               symbol=best_symbol, order_type=order_dir)

        # Change 2: herd agents pile on — cross the spread when signal is strong
        if best_imbalance > 0:
            if conviction > 0.4:
                best_ask  = exchange.order_book.get_best_ask(best_symbol)
                bid_price = best_ask.price * 1.001 if best_ask else price * 1.005
            else:
                bid_price = price * 1.005
            return [Order(self.agent_id, best_symbol, OrderType.BUY,
                          bid_price, quantity, exchange.step, ttl=4)]
        else:
            if conviction > 0.4:
                best_bid  = exchange.order_book.get_best_bid(best_symbol)
                ask_price = best_bid.price * 0.999 if best_bid else price * 0.995
            else:
                ask_price = price * 0.995
            return [Order(self.agent_id, best_symbol, OrderType.SELL,
                          ask_price, quantity, exchange.step, ttl=4)]

class ArbitrageAgent(BaseAgent):
    agent_type = AgentType.ARBITRAGE

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 30,
                 threshold:     float = 1.5,    # z-score entry threshold
                 risk_aversion: float = 0.8,
                 cash_preference: float = 0.8,
                 max_ownership_pct: float = 0.30):
        super().__init__(agent_id, initial_cash, cash_preference, max_ownership_pct)
        self.lookback      = lookback
        self.threshold     = threshold
        self.risk_aversion = risk_aversion
        # ratio_history[(sym_a, sym_b)] = list of log-ratio floats
        self._ratio_history: dict[tuple[str, str], list[float]] = {}

    def _update_ratios(self, exchange: Exchange, symbols: list[str]) -> None:

        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                pa = exchange.get_price(a)
                pb = exchange.get_price(b)
                if pa and pb and pa > 0 and pb > 0:
                    ratio = math.log(pa / pb)
                    key   = (a, b)
                    if key not in self._ratio_history:
                        self._ratio_history[key] = []
                    self._ratio_history[key].append(ratio)
                    # keep only the rolling window
                    if len(self._ratio_history[key]) > self.lookback:
                        self._ratio_history[key].pop(0)

    def decide(self, exchange: Exchange) -> list[Order]:
        symbols = list(exchange.order_book.stocks.keys())
        if len(symbols) < 2:
            return []

        self._update_ratios(exchange, symbols)

        best_z     = 0.0
        best_orders: list[Order] = []

        for (sym_a, sym_b), history in self._ratio_history.items():
            if len(history) < max(5, self.lookback // 2):
                continue

            mean  = statistics.mean(history)
            stdev = statistics.stdev(history) if len(history) > 1 else 0.0
            if stdev < 1e-10:
                continue

            current_ratio = history[-1]
            z             = (current_ratio - mean) / stdev

            if abs(z) <= self.threshold or abs(z) <= abs(best_z):
                continue

            pa = exchange.get_price(sym_a)
            pb = exchange.get_price(sym_b)
            if pa is None or pb is None:
                continue

            conviction = min(abs(z) / (self.threshold * 2), 1.0)
            best_z     = z

            # Change 3: submit both legs simultaneously — true pairs trade
            if z > 0:
                # A expensive vs B → sell A, buy B
                qty_a = self.calculate_order_size(conviction, pa,
                                                  self.risk_aversion, exchange,
                                                  symbol=sym_a, order_type=OrderType.SELL)
                qty_b = self.calculate_order_size(conviction, pb,
                                                  self.risk_aversion, exchange,
                                                  symbol=sym_b, order_type=OrderType.BUY)
                best_bid_a = exchange.order_book.get_best_bid(sym_a)
                best_ask_b = exchange.order_book.get_best_ask(sym_b)
                # Change 2: cross the spread on both legs for immediate fill
                ask_a = best_bid_a.price * 0.999 if best_bid_a else pa * 0.995
                bid_b = best_ask_b.price * 1.001 if best_ask_b else pb * 1.005
                best_orders = [
                    Order(self.agent_id, sym_a, OrderType.SELL,
                          ask_a, qty_a, exchange.step, ttl=5),
                    Order(self.agent_id, sym_b, OrderType.BUY,
                          bid_b, qty_b, exchange.step, ttl=5),
                ]
            else:
                # A cheap vs B → buy A, sell B
                qty_a = self.calculate_order_size(conviction, pa,
                                                  self.risk_aversion, exchange,
                                                  symbol=sym_a, order_type=OrderType.BUY)
                qty_b = self.calculate_order_size(conviction, pb,
                                                  self.risk_aversion, exchange,
                                                  symbol=sym_b, order_type=OrderType.SELL)
                best_ask_a = exchange.order_book.get_best_ask(sym_a)
                best_bid_b = exchange.order_book.get_best_bid(sym_b)
                bid_a = best_ask_a.price * 1.001 if best_ask_a else pa * 1.005
                ask_b = best_bid_b.price * 0.999 if best_bid_b else pb * 0.995
                best_orders = [
                    Order(self.agent_id, sym_a, OrderType.BUY,
                          bid_a, qty_a, exchange.step, ttl=5),
                    Order(self.agent_id, sym_b, OrderType.SELL,
                          ask_b, qty_b, exchange.step, ttl=5),
                ]

        return best_orders