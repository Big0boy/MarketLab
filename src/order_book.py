import models
from models import OrderType
from collections import defaultdict, deque


class OrderBook:

    def __init__(self):
        # symbol -> price -> queue of orders (FIFO)
        self.bids = defaultdict(lambda: defaultdict(deque))
        self.asks = defaultdict(lambda: defaultdict(deque))
        self.stocks = {}

    def add_order(self, order: models.Order) -> None:
        book = self.bids if order.order_type == OrderType.BUY else self.asks
        book[order.symbol][order.price].append(order)

    def get_best_bid(self, symbol: str) -> models.Order | None:
        if symbol not in self.bids or not self.bids[symbol]:
            return None
        best_price = max(self.bids[symbol].keys())
        return self.bids[symbol][best_price][0]

    def get_best_ask(self, symbol: str) -> models.Order | None:
        if symbol not in self.asks or not self.asks[symbol]:
            return None
        best_price = min(self.asks[symbol].keys())
        return self.asks[symbol][best_price][0]

    def pop_best_bid(self, symbol: str) -> models.Order | None:
        best = self.get_best_bid(symbol)
        if not best:
            return None
        q = self.bids[symbol][best.price]
        order = q.popleft()
        if not q:
            del self.bids[symbol][best.price]
        return order

    def pop_best_ask(self, symbol: str) -> models.Order | None:
        best = self.get_best_ask(symbol)
        if not best:
            return None
        q = self.asks[symbol][best.price]
        order = q.popleft()
        if not q:
            del self.asks[symbol][best.price]
        return order

    def get_all_symbols(self) -> list[str]:
        return list(set(self.bids.keys()) | set(self.asks.keys()))

    def get_mid_price(self, symbol: str) -> float | None:
        bid = self.get_best_bid(symbol)
        ask = self.get_best_ask(symbol)
        if bid and ask:
            return (bid.price + ask.price) / 2
        return None

    def get_spread(self, symbol: str) -> float | None:
        bid = self.get_best_bid(symbol)
        ask = self.get_best_ask(symbol)
        if bid and ask:
            return round(ask.price - bid.price, 4)
        return None

    def remove_order(self, order: models.Order) -> None:
        book = self.bids if order.order_type == OrderType.BUY else self.asks
        levels = book.get(order.symbol, {})
        queue = levels.get(order.price)

        if queue and order in queue:
            queue.remove(order)
            if not queue:
                del levels[order.price]

    def expire_orders(self, current_step: int) -> list:
        expired = []
        for book in [self.bids, self.asks]:
            for symbol in list(book.keys()):
                for price in list(book[symbol].keys()):
                    queue = book[symbol][price]
                    fresh = deque()
                    for o in queue:
                        age = current_step - o.timestamp
                        if o.ttl > 0 and age > o.ttl:
                            expired.append(o)
                        else:
                            fresh.append(o)
                    if fresh:
                        book[symbol][price] = fresh
                    else:
                        del book[symbol][price]
        return expired

    def remove_agent_orders(self, agent_id: int) -> list:
        removed = []
        for book in [self.bids, self.asks]:
            for symbol in list(book.keys()):
                for price in list(book[symbol].keys()):
                    queue = book[symbol][price]
                    kept  = deque()
                    for o in queue:
                        if o.agent_id == agent_id:
                            removed.append(o)
                        else:
                            kept.append(o)
                    if kept:
                        book[symbol][price] = kept
                    else:
                        del book[symbol][price]
        return removed