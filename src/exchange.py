import random
import models
from models import OrderType
from order_book import OrderBook
from cda import CDA


class Exchange:

    def __init__(self, stocks: list[models.Stock], slippage_factor: float = 0.0001):
        self.order_book = OrderBook()
        self.cda        = CDA(self.order_book, slippage_factor)

        for stock in stocks:
            self.order_book.stocks[stock.symbol] = stock

        self.agents    = {}
        self.trade_log = []
        self.step      = 0

    def register_agent(self, agent) -> None:
        self.agents[agent.agent_id] = agent

    def submit_order(self, order: models.Order) -> None:
        agent = self.agents.get(order.agent_id)
        if not agent:
            print(f"Agent {order.agent_id} not found.")
            return

        if order.quantity <= 0 or order.price <= 0:
            print(f"Agent {order.agent_id} placed invalid order (qty/price <= 0).")
            return

        if order.order_type == OrderType.BUY:
            if agent.cash >= order.price * order.quantity:
                self.order_book.add_order(order)
            else:
                print(f"Agent {order.agent_id} has insufficient cash.")

        elif order.order_type == OrderType.SELL:
            if agent.holdings.get(order.symbol, 0) >= order.quantity:
                self.order_book.add_order(order)
            else:
                print(f"Agent {order.agent_id} has insufficient holdings.")

    def run_step(self) -> list[models.Trade]:
        trades = self.cda.run_auction(self.step)
        for trade in trades:
            self._settle_trade(trade)
        self._apply_news_shock()
        for symbol in list(self.order_book.stocks.keys()):
            self._check_circuit_breaker(symbol)

        self.order_book.clear()

        self.step += 1

        return trades

    def _settle_trade(self, trade: models.Trade) -> None:
        buyer  = self.agents[trade.buyer_id]
        seller = self.agents[trade.seller_id]
        total_cost = trade.price * trade.quantity

        buyer.cash                          -= total_cost
        buyer.holdings[trade.symbol]         = buyer.holdings.get(trade.symbol, 0) + trade.quantity

        seller.cash                         += total_cost
        seller.holdings[trade.symbol]        = seller.holdings.get(trade.symbol, 0) - trade.quantity

        self.trade_log.append(trade)

    def get_price(self, symbol: str) -> float | None:
        mid   = self.cda.calculate_price(symbol)
        if mid:
            return mid
        stock = self.order_book.stocks.get(symbol)
        return stock.price if stock else None

    def get_fundamental_value(self, symbol: str) -> float | None:
        stock = self.order_book.stocks.get(symbol)
        return stock.fundamental_value if stock else None

    def get_price_history(self, symbol: str) -> list[float] | None:
        stock = self.order_book.stocks.get(symbol)
        return stock.price_history if stock else None

    def get_spread(self, symbol: str) -> float | None:
        return self.order_book.get_spread(symbol)

    def _apply_news_shock(self) -> None:
        for stock in self.order_book.stocks.values():
            if random.random() >0.02:
                continue
            shock = random.gauss(0, 0.001)       
            stock.fundamental_value *= (1 + shock)

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
            print(f"Circuit breaker triggered for {symbol} "
                  f"at step {self.step} — price change: {change:.2%}")

    def get_trade_log(self) -> list[models.Trade]:
        return self.trade_log

    def get_market_summary(self) -> dict:
        summary = {}
        for symbol in self.order_book.stocks.keys():
            price       = self.get_price(symbol)
            fundamental = self.get_fundamental_value(symbol)
            summary[symbol] = {
                "price"          : price,
                "fundamental"    : fundamental,
                "spread"         : self.get_spread(symbol),
                "total_trades"   : len([t for t in self.trade_log if t.symbol == symbol]),
                "price_change_%" : round((price - fundamental) / fundamental * 100, 2)
                                   if price and fundamental else None
            }
        return summary