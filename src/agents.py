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