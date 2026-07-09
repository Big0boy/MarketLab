import math
import models
from models import OrderType


def _snap_to_tick(price: float, tick_size: float) -> float:
    """Feature 3: round price to nearest tick increment."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


class CDA:

    def __init__(self, ob, slippage_factor: float = 0.0001):
        self.order_book     = ob
        self.slippage_factor = slippage_factor
        # Feature 1: per-symbol alternating side flag for bid-ask bounce
        self._last_side: dict[str, OrderType] = {}

    def calculate_price(self, symbol: str) -> float | None:
        return self.order_book.get_mid_price(symbol)

    def _apply_slippage(self, symbol: str, quantity: int) -> None:
        """Feature 2: sqrt-volume market impact instead of linear."""
        stock = self.order_book.stocks[symbol]
        adv   = max(stock.adv, 1.0)
        # Almgren-style: impact ∝ sqrt(qty / adv)
        impact = self.slippage_factor * math.sqrt(quantity / adv)
        stock.price *= (1 + impact)
        stock.price = max(stock.price, 0.01)

    def _trade_price(self, symbol: str, bid: models.Order,
                     ask: models.Order) -> float:
        """Feature 1: alternate between ask and bid price each trade
        to produce realistic bid-ask bounce in the return series."""
        last = self._last_side.get(symbol, OrderType.SELL)
        if last == OrderType.SELL:
            # last trade was at ask → this one at bid
            price = bid.price
            self._last_side[symbol] = OrderType.BUY
        else:
            # last trade was at bid → this one at ask
            price = ask.price
            self._last_side[symbol] = OrderType.SELL

        # Feature 3: snap to tick
        stock = self.order_book.stocks.get(symbol)
        if stock:
            price = _snap_to_tick(price, stock.tick_size)
        return price

    def execute_trade(self, symbol: str, timestamp: int,
                      volume_multiplier: float = 1.0):
        """Execute all crossing orders for one symbol.

        Feature 5: volume_multiplier scales trade sizes to model
        intraday U-shaped volume patterns.
        """
        trades = []
        step_volume = 0

        while True:
            bid = self.order_book.get_best_bid(symbol)
            ask = self.order_book.get_best_ask(symbol)

            if not bid or not ask or bid.price < ask.price:
                break

            bid = self.order_book.pop_best_bid(symbol)
            ask = self.order_book.pop_best_ask(symbol)

            # Feature 5: scale matched quantity by intraday volume multiplier
            base_qty  = min(bid.quantity, ask.quantity)
            trade_qty = max(1, int(base_qty * volume_multiplier))

            # Feature 1+3: alternating bid/ask bounce + tick-snapped price
            trade_price = self._trade_price(symbol, bid, ask)

            trade = models.Trade(
                buyer_id  = bid.agent_id,
                seller_id = ask.agent_id,
                symbol    = symbol,
                price     = trade_price,
                quantity  = trade_qty,
                timestamp = timestamp,
            )

            # Partial fill: put remainder back
            if bid.quantity > ask.quantity:
                bid.quantity -= trade_qty
                if bid.quantity > 0:
                    self.order_book.add_order(bid)
            elif ask.quantity > bid.quantity:
                ask.quantity -= trade_qty
                if ask.quantity > 0:
                    self.order_book.add_order(ask)

            # Feature 2: sqrt-volume slippage (buy pressure moves price up)
            self._apply_slippage(symbol, trade_qty)

            step_volume += trade_qty
            trades.append(trade)

        # Feature 2: update rolling ADV (exponential moving average, α=0.1)
        stock = self.order_book.stocks.get(symbol)
        if stock:
            stock.adv = 0.9 * stock.adv + 0.1 * step_volume

        return trades

    def run_auction(self, timestamp: int,
                    skip_symbols: set[str] | None = None,
                    volume_multiplier: float = 1.0):
        # Change 3: skip auctioning halted stocks
        skip_symbols = skip_symbols or set()
        trades = []
        for symbol in self.order_book.get_all_symbols():
            if symbol in skip_symbols:
                continue
            trades.extend(
                self.execute_trade(symbol, timestamp, volume_multiplier)
            )
        return trades