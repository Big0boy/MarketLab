from __future__ import annotations
import random
from typing import TYPE_CHECKING
from models import Order, OrderType, AgentType

if TYPE_CHECKING:
    from exchange import Exchange


class BaseAgent:

    agent_type: AgentType = AgentType.Base

    def __init__(self, agent_id: int, initial_cash: float = 10000.0):
        self.agent_id       = agent_id
        self.cash           = initial_cash
        self.holdings       = {}
        self.trade_history  = []
        self.wealth_history = []
        self.raw_fitness = 0
        self.fitness = 0

    def decide(self, exchange: Exchange) -> Order | None:
        return None

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
                             exchange: Exchange) -> int:
        wealth   = self.get_total_wealth(exchange)
        raw_size = (conviction * wealth) / (price * risk_aversion)
        return max(1, min(int(raw_size), 300))

    def __repr__(self) -> str:
        return (f"{self.agent_type.value}("
                f"id={self.agent_id}, "
                f"cash={self.cash:.2f}, "
                f"holdings={self.holdings})")

# FUNDAMENTALIST
class FundamentalistAgent(BaseAgent):
    agent_type = AgentType.FUNDAMENTALIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 buy_threshold:  float = 0.98,
                 sell_threshold: float = 1.02,
                 risk_aversion:  float = 1.0):
        super().__init__(agent_id, initial_cash)
        self.buy_threshold  = buy_threshold   
        self.sell_threshold = sell_threshold  
        self.risk_aversion  = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        best_order     = None
        best_deviation = 0.0

        for symbol in exchange.order_book.stocks:
            price       = exchange.get_price(symbol)
            fundamental = exchange.get_fundamental_value(symbol)
            if price is None or fundamental is None:
                continue

            deviation  = abs(price - fundamental) / fundamental
            sediment   = exchange.order_book.stocks[symbol].sediment
            buy_conviction  = min(deviation * 2 * (1 + max(sediment, 0)), 1.0)
            sell_conviction = min(deviation * 2 * (1 + max(-sediment, 0)), 1.0)

            if deviation > best_deviation:
                if price < fundamental * self.buy_threshold:
                    quantity       = self.calculate_order_size(buy_conviction, price,
                                                               self.risk_aversion, exchange)
                    best_order     = Order(self.agent_id, symbol, OrderType.BUY,
                                          price * 1.005, quantity, exchange.step)
                    best_deviation = deviation
                elif price > fundamental * self.sell_threshold:
                    quantity       = self.calculate_order_size(sell_conviction, price,
                                                               self.risk_aversion, exchange)
                    best_order     = Order(self.agent_id, symbol, OrderType.SELL,
                                          price * 0.995, quantity, exchange.step)
                    best_deviation = deviation

        return best_order


#CHARTIST
class ChartistAgent(BaseAgent):
    agent_type = AgentType.CHARTIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 5,
                 risk_aversion: float = 1.2):
        super().__init__(agent_id, initial_cash)
        self.lookback      = lookback
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        best_order = None
        best_trend = 0.0

        for symbol in exchange.order_book.stocks:
            history = exchange.get_price_history(symbol)
            if not history or len(history) < self.lookback:
                continue

            price_now  = history[-1]
            price_then = history[-self.lookback]
            if price_then == 0:
                continue

            sediment   = exchange.order_book.stocks[symbol].sediment
            raw_trend  = (price_now - price_then) / price_then
            # Sediment in same direction as trend → amplify
            # Sediment against trend → dampen (market reversing)
            aligned    = (raw_trend >= 0 and sediment >= 0) or (raw_trend < 0 and sediment < 0)
            alignment  = (1 + sediment) if aligned else (1 - abs(sediment) * 0.5)
            trend      = raw_trend * max(alignment, 0.1)
            conviction = min(abs(trend) * 5, 1.0)
            quantity   = self.calculate_order_size(conviction, price_now,
                                                   self.risk_aversion, exchange)

            if abs(trend) > abs(best_trend):
                if trend > 0:
                    best_order = Order(self.agent_id, symbol, OrderType.BUY,
                                       price_now * 1.005, quantity, exchange.step)
                    best_trend = trend
                elif trend < 0:
                    best_order = Order(self.agent_id, symbol, OrderType.SELL,
                                       price_now * 0.995, quantity, exchange.step)
                    best_trend = trend

        return best_order


# NOISE TRADER
class NoiseAgent(BaseAgent):
    agent_type = AgentType.NOISE

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 hold_prob:float = 0.40,
                 buy_prob:float = 0.30,
                 risk_aversion: float = 2.0):
        super().__init__(agent_id, initial_cash)
        self.hold_prob     = hold_prob
        self.buy_prob      = buy_prob
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        if random.random() < self.hold_prob:
            return None

        symbols = list(exchange.order_book.stocks.keys())
        if not symbols:
            return None

        symbol = random.choice(symbols)
        price  = exchange.get_price(symbol)
        if price is None:
            return None

        sediment   = exchange.order_book.stocks[symbol].sediment
        conviction = random.uniform(0.1, 0.4) * (1 + abs(sediment))
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange)
        roll = random.random()
        buy_threshold = self.buy_prob * (1 + sediment * 0.5)
        if roll < buy_threshold:
            return Order(self.agent_id, symbol, OrderType.BUY,
                         price * 1.005, quantity, exchange.step)
        else:
            return Order(self.agent_id, symbol, OrderType.SELL,
                         price * 0.995, quantity, exchange.step)


# MARKET MAKER
class MarketMakerAgent(BaseAgent):
    agent_type = AgentType.MARKET_MAKER

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 spread_pct:    float = 0.004,
                 risk_aversion: float = 0.5):
        super().__init__(agent_id, initial_cash)
        self.spread_pct    = spread_pct
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        symbols = list(exchange.order_book.stocks.keys())
        n       = len(symbols) or 1

        for symbol in symbols:
            price = exchange.get_price(symbol)
            if price is None:
                continue

            sediment   = exchange.order_book.stocks[symbol].sediment
            # Market makers profit from volume regardless of direction
            # so abs(sediment) is correct — more volatility = more quotes
            conviction = 0.3 * (1 + abs(sediment))
            quantity   = self.calculate_order_size(conviction, price,
                                                   self.risk_aversion, exchange)

            bid_price = round(price * (1 - self.spread_pct / 2), 4)
            ask_price = round(price * (1 + self.spread_pct / 2), 4)

            # Size bid so we don't over-commit cash across all symbols
            affordable = max(1, int(self.cash / (bid_price * n)))
            bid_qty    = min(quantity, affordable)

            if self.cash >= bid_price * bid_qty:
                exchange.submit_order(Order(self.agent_id, symbol, OrderType.BUY,
                                            bid_price, bid_qty, exchange.step))

            if self.holdings.get(symbol, 0) - quantity >= -exchange.max_short:
                exchange.submit_order(Order(self.agent_id, symbol, OrderType.SELL,
                                            ask_price, quantity, exchange.step))

        return None


class MomentumAgent(BaseAgent):
    agent_type = AgentType.MOMENTUM

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 10,
                 threshold:     float = 0.005,
                 risk_aversion: float = 1.5):
        super().__init__(agent_id, initial_cash)
        self.lookback      = lookback       
        self.threshold     = threshold      
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        best_symbol    = None
        best_signal    = 0.0   

        for symbol in exchange.order_book.stocks:
            history = exchange.get_price_history(symbol)
            if not history or len(history) < self.lookback:
                continue

            price_now  = history[-1]
            price_then = history[-self.lookback]
            if price_then == 0:
                continue

            momentum = (price_now - price_then) / price_then
            if abs(momentum) > abs(best_signal) and abs(momentum) > self.threshold:
                best_signal = momentum
                best_symbol = symbol

        if best_symbol is None:
            return None

        price      = exchange.get_price(best_symbol)
        if price is None:
            return None

        conviction = min(abs(best_signal) * 10, 1.0)
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange)

        if best_signal > 0:
            return Order(self.agent_id, best_symbol, OrderType.BUY,
                         price * 1.005, quantity, exchange.step)
        else:
            return Order(self.agent_id, best_symbol, OrderType.SELL,
                         price * 0.995, quantity, exchange.step)


class MeanReversionAgent(BaseAgent):
    agent_type = AgentType.MEAN_REVERSION

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 20,
                 entry_z:       float = 0.01,
                 risk_aversion: float = 1.2):
        super().__init__(agent_id, initial_cash)
        self.lookback      = lookback    # rolling mean window
        self.entry_z       = entry_z     # minimum deviation fraction to act
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
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

            deviation = (price_now - mean) / mean   # signed: + means above mean

            if abs(deviation) > abs(best_deviation) and abs(deviation) > self.entry_z:
                best_deviation = deviation
                best_symbol    = symbol
                # Price above mean → sell (expect reversion down)
                # Price below mean → buy  (expect reversion up)
                best_direction = OrderType.SELL if deviation > 0 else OrderType.BUY

        if best_symbol is None:
            return None

        price = exchange.get_price(best_symbol)
        if price is None:
            return None

        conviction = min(abs(best_deviation) * 20, 1.0)
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange)

        if best_direction == OrderType.BUY:
            return Order(self.agent_id, best_symbol, OrderType.BUY,
                         price * 1.005, quantity, exchange.step)
        else:
            return Order(self.agent_id, best_symbol, OrderType.SELL,
                         price * 0.995, quantity, exchange.step)


class HerdAgent(BaseAgent):
    agent_type = AgentType.HERD

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 sensitivity:   float = 1.0,
                 risk_aversion: float = 1.8):
        super().__init__(agent_id, initial_cash)
        self.sensitivity   = sensitivity  
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        best_symbol    = None
        best_imbalance = 0.0

        for symbol in exchange.order_book.stocks:
            bids = exchange.order_book.bids.get(symbol, [])
            asks = exchange.order_book.asks.get(symbol, [])

            total_bid_qty = sum(o.quantity for o in bids)
            total_ask_qty = sum(o.quantity for o in asks)
            total         = total_bid_qty + total_ask_qty
            if total == 0:
                continue


            imbalance = (total_bid_qty - total_ask_qty) / total

            # Sediment reinforces the observed crowd direction
            sediment  = exchange.order_book.stocks[symbol].sediment
            signal    = imbalance * (1 + abs(sediment) * self.sensitivity)
            signal    = max(-1.0, min(signal, 1.0))

            if abs(signal) > abs(best_imbalance):
                best_imbalance = signal
                best_symbol    = symbol

        if best_symbol is None or abs(best_imbalance) < 0.05:
            return None

        price = exchange.get_price(best_symbol)
        if price is None:
            return None

        conviction = min(abs(best_imbalance), 1.0)
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange)

        if best_imbalance > 0:
            return Order(self.agent_id, best_symbol, OrderType.BUY,
                         price * 1.005, quantity, exchange.step)
        else:
            return Order(self.agent_id, best_symbol, OrderType.SELL,
                         price * 0.995, quantity, exchange.step)

class ArbitrageAgent(BaseAgent):
    """
    Pairs / statistical-arbitrage agent.

    For every pair of stocks (A, B) the agent tracks the log-price ratio
    r_t = log(P_A / P_B) over a rolling window.  When r_t deviates more
    than `threshold` standard deviations from its rolling mean it bets on
    mean-reversion:
      - ratio too HIGH  → sell the expensive leg (A), buy the cheap leg (B)
      - ratio too LOW   → buy the expensive leg (A), sell the cheap leg (B)

    The agent alternates legs each step to avoid submitting two orders at once
    (exchange accepts one order per agent per step).
    """
    agent_type = AgentType.ARBITRAGE

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 30,
                 threshold:     float = 1.5,    # z-score entry threshold
                 risk_aversion: float = 0.8):
        super().__init__(agent_id, initial_cash)
        self.lookback      = lookback
        self.threshold     = threshold
        self.risk_aversion = risk_aversion
        # ratio_history[(sym_a, sym_b)] = list of log-ratio floats
        self._ratio_history: dict[tuple[str, str], list[float]] = {}

    def _update_ratios(self, exchange: Exchange, symbols: list[str]) -> None:
        """Append the current log-ratio for every ordered pair."""
        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                pa = exchange.get_price(a)
                pb = exchange.get_price(b)
                if pa and pb and pa > 0 and pb > 0:
                    import math
                    ratio = math.log(pa / pb)
                    key   = (a, b)
                    if key not in self._ratio_history:
                        self._ratio_history[key] = []
                    self._ratio_history[key].append(ratio)
                    # keep only the rolling window
                    if len(self._ratio_history[key]) > self.lookback:
                        self._ratio_history[key].pop(0)

    def decide(self, exchange: Exchange) -> Order | None:
        symbols = list(exchange.order_book.stocks.keys())
        if len(symbols) < 2:
            return None

        self._update_ratios(exchange, symbols)

        best_z       = 0.0
        best_order   = None

        for (sym_a, sym_b), history in self._ratio_history.items():
            if len(history) < max(5, self.lookback // 2):
                continue  # not enough data yet

            import statistics
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

            if z > 0:
                # A is expensive relative to B → sell A (even steps), buy B (odd steps)
                if exchange.step % 2 == 0:
                    qty = self.calculate_order_size(conviction, pa,
                                                    self.risk_aversion, exchange)
                    best_order = Order(self.agent_id, sym_a, OrderType.SELL,
                                       pa * 0.995, qty, exchange.step)
                else:
                    qty = self.calculate_order_size(conviction, pb,
                                                    self.risk_aversion, exchange)
                    best_order = Order(self.agent_id, sym_b, OrderType.BUY,
                                       pb * 1.005, qty, exchange.step)
            else:
                # A is cheap relative to B → buy A (even steps), sell B (odd steps)
                if exchange.step % 2 == 0:
                    qty = self.calculate_order_size(conviction, pa,
                                                    self.risk_aversion, exchange)
                    best_order = Order(self.agent_id, sym_a, OrderType.BUY,
                                       pa * 1.005, qty, exchange.step)
                else:
                    qty = self.calculate_order_size(conviction, pb,
                                                    self.risk_aversion, exchange)
                    best_order = Order(self.agent_id, sym_b, OrderType.SELL,
                                       pb * 0.995, qty, exchange.step)

        return best_order