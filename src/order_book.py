import models
from models import OrderType


class OrderBook:

    def __init__(self):
        self.bids   = {}       
        self.asks   = {}       
        self.stocks = {}       

    def add_order(self, order: models.Order) -> None:
        if order.order_type == OrderType.BUY:
            if order.symbol not in self.bids:
                self.bids[order.symbol] = []
            self.bids[order.symbol].append(order)

        elif order.order_type == OrderType.SELL:
            if order.symbol not in self.asks:
                self.asks[order.symbol] = []
            self.asks[order.symbol].append(order)

    def get_best_bid(self, symbol: str) -> models.Order | None:
        if symbol in self.bids and self.bids[symbol]:
            return max(self.bids[symbol], key=lambda o: o.price)
        return None

    def get_best_ask(self, symbol: str) -> models.Order | None:
        if symbol in self.asks and self.asks[symbol]:
            return min(self.asks[symbol], key=lambda o: o.price)
        return None

    def get_mid_price(self, symbol: str) -> float | None:
        best_bid = self.get_best_bid(symbol)
        best_ask = self.get_best_ask(symbol)
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

    def get_spread(self, symbol: str) -> float | None:
        best_bid = self.get_best_bid(symbol)
        best_ask = self.get_best_ask(symbol)
        if best_bid and best_ask:
            return round(abs(best_ask.price - best_bid.price), 4)
        return None

    def get_all_symbols(self) -> list[str]:
        return list(set(list(self.bids.keys()) + list(self.asks.keys())))

    def remove_order(self, order: models.Order) -> None:
        if order.order_type == OrderType.BUY:
            if order.symbol in self.bids:
                self.bids[order.symbol].remove(order)

        elif order.order_type == OrderType.SELL:
            if order.symbol in self.asks:
                self.asks[order.symbol].remove(order)

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()