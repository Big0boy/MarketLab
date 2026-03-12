import order_book
import models
from models import OrderType


class CDA:

    def __init__(self, ob: order_book.OrderBook, slippage_factor: float = 0.0001):
        self.order_book      = ob
        self.slippage_factor = slippage_factor

    def calculate_price(self, symbol: str) -> float | None:
        bid = self.order_book.get_best_bid(symbol)
        ask = self.order_book.get_best_ask(symbol)
        if bid and ask:
            return (bid.price + ask.price) / 2
        return None

    def _apply_slippage(self, symbol: str, order_type: OrderType, quantity: int) -> None:
        stock  = self.order_book.stocks[symbol]
        impact = quantity * self.slippage_factor

        if order_type == OrderType.BUY:
            stock.price = stock.price * (1 + impact)    # buying pushes price UP
        elif order_type == OrderType.SELL:
            stock.price = stock.price * (1 - impact)    # selling pushes price DOWN

        stock.price_history.append(round(stock.price, 4))

    def _handle_partial_fills(self, bid: models.Order, ask: models.Order, trade_quantity: int) -> None:
        if bid.quantity > ask.quantity:
            bid.quantity -= trade_quantity              # bid partially filled → stays
            self.order_book.remove_order(ask)           # ask fully filled   → removed

        elif ask.quantity > bid.quantity:
            ask.quantity -= trade_quantity              # ask partially filled → stays
            self.order_book.remove_order(bid)           # bid fully filled    → removed

        else:
            self.order_book.remove_order(bid)           # both fully filled   → both removed
            self.order_book.remove_order(ask)

    def execute_trade(self, symbol: str, timestamp: int) -> list[models.Trade]:
        trades = []

        while True:
            bid = self.order_book.get_best_bid(symbol)
            ask = self.order_book.get_best_ask(symbol)

            # stop if no orders or prices don't cross
            if not bid or not ask or bid.price < ask.price:
                break

            trade_price    = self.calculate_price(symbol)
            trade_quantity = min(bid.quantity, ask.quantity)

            trade = models.Trade(
                buyer_id  = bid.agent_id,
                seller_id = ask.agent_id,
                symbol    = symbol,
                price     = trade_price,
                quantity  = trade_quantity,
                timestamp = timestamp
            )

            self._handle_partial_fills(bid, ask, trade_quantity)
            self._apply_slippage(symbol, OrderType.BUY, trade_quantity)

            trades.append(trade)

        return trades

    def run_auction(self, timestamp: int) -> list[models.Trade]:
        all_trades = []
        for symbol in self.order_book.get_all_symbols():
            trades = self.execute_trade(symbol, timestamp)
            all_trades.extend(trades)
        return all_trades