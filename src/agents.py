import random
from __future__ import annotations
from typing import TYPE_CHECKING
from models import Order, OrderType, AgentType

if TYPE_CHECKING:
    from exchange import Exchange


class BaseAgent:

    agent_type: AgentType = AgentType.BASE  # subclasses override this

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
        return max(1, min(int(raw_size), 100))

    def __repr__(self) -> str:
        return (f"{self.agent_type.value}("
                f"id={self.agent_id}, "
                f"cash={self.cash:.2f}, "
                f"holdings={self.holdings})")

# FUNDAMENTALIST
class FundamentalistAgent(BaseAgent):
    agent_type = AgentType.FUNDAMENTALIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 buy_threshold:  float = 0.95,
                 sell_threshold: float = 1.05,
                 risk_aversion:  float = 1.0):
        super().__init__(agent_id, initial_cash)
        self.buy_threshold  = buy_threshold   # buy if price < fundamental × 0.95
        self.sell_threshold = sell_threshold  # sell if price > fundamental × 1.05
        self.risk_aversion  = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        for symbol in exchange.order_book.stocks:

            price       = exchange.get_price(symbol)
            fundamental = exchange.get_fundamental_value(symbol)

            if price is None or fundamental is None:
                continue

            deviation  = abs(price - fundamental) / fundamental
            conviction = min(deviation * 2, 1.0)

            quantity = self.calculate_order_size(conviction, price,
                                                 self.risk_aversion, exchange)

            if price < fundamental * self.buy_threshold:
                return Order(self.agent_id, symbol, OrderType.BUY,
                             price, quantity, exchange.step)

            if price > fundamental * self.sell_threshold and self.holdings.get(symbol, 0) > 0:
                return Order(self.agent_id, symbol, OrderType.SELL,
                             price, quantity, exchange.step)

        return None 

# CHARTIST
class ChartistAgent(BaseAgent):
    agent_type = AgentType.CHARTIST

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 lookback:      int   = 5,
                 risk_aversion: float = 1.0):
        super().__init__(agent_id, initial_cash)
        self.lookback      = lookback       # how many past prices to look at
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        for symbol in exchange.order_book.stocks:

            history = exchange.get_price_history(symbol)

            # need enough history to detect a trend
            if not history or len(history) < self.lookback:
                continue

            price_now  = history[-1]
            price_then = history[-self.lookback]

            if price_then == 0:
                continue

            # how strong is the trend?
            trend      = (price_now - price_then) / price_then
            conviction = min(abs(trend) * 5, 1.0)   # stronger trend → more conviction

            quantity = self.calculate_order_size(conviction, price_now,
                                                 self.risk_aversion, exchange)

            if trend > 0:
                return Order(self.agent_id, symbol, OrderType.BUY,
                             price_now, quantity, exchange.step)

            if trend < 0 and self.holdings.get(symbol, 0) > 0:
                return Order(self.agent_id, symbol, OrderType.SELL,
                             price_now, quantity, exchange.step)

        return None   

# NOISE TRADER
class NoiseAgent(BaseAgent):
    agent_type = AgentType.NOISE

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 hold_prob: float = 0.70,
                 buy_prob:  float = 0.15,
                 risk_aversion: float = 2.0):  
        super().__init__(agent_id, initial_cash)
        self.hold_prob     = hold_prob
        self.buy_prob      = buy_prob
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        roll = random.random()

        if roll < self.hold_prob:
            return None

        symbols = list(exchange.order_book.stocks.keys())
        if not symbols:
            return None

        symbol = random.choice(symbols)
        price  = exchange.get_price(symbol)
        if price is None:
            return None

        conviction = random.uniform(0.1, 0.4)
        quantity   = self.calculate_order_size(conviction, price,
                                               self.risk_aversion, exchange)

        if roll < self.hold_prob + self.buy_prob:
            return Order(self.agent_id, symbol, OrderType.BUY,
                         price, quantity, exchange.step)

        if self.holdings.get(symbol, 0) > 0:   # only sell if holding something
            return Order(self.agent_id, symbol, OrderType.SELL,
                         price, quantity, exchange.step)

        return None

# MARKET MAKER
class MarketMakerAgent(BaseAgent):
    agent_type = AgentType.MARKET_MAKER

    def __init__(self, agent_id: int, initial_cash: float = 10000.0,
                 spread_pct:    float = 0.02,    
                 risk_aversion: float = 0.5):
        super().__init__(agent_id, initial_cash)
        self.spread_pct    = spread_pct
        self.risk_aversion = risk_aversion

    def decide(self, exchange: Exchange) -> Order | None:
        orders = []

        for symbol in exchange.order_book.stocks:
            price = exchange.get_price(symbol)
            if price is None:
                continue

            conviction = 0.3 
            quantity   = self.calculate_order_size(conviction, price,
                                                   self.risk_aversion, exchange)

            bid_price = round(price * (1 - self.spread_pct / 2), 4)  # slightly below
            ask_price = round(price * (1 + self.spread_pct / 2), 4)  # slightly above

            if self.cash >= bid_price * quantity:
                orders.append(Order(self.agent_id, symbol, OrderType.BUY,
                                    bid_price, quantity, exchange.step))

            if self.holdings.get(symbol, 0) >= quantity:
                orders.append(Order(self.agent_id, symbol, OrderType.SELL,
                                    ask_price, quantity, exchange.step))

        for order in orders[1:]:
            exchange.submit_order(order)

        return orders[0] if orders else None