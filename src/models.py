from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    agent_id:   int
    symbol:     str
    order_type: OrderType   
    price:      float       
    quantity:   int         
    timestamp:  int         


@dataclass
class Trade:
    buyer_id:   int
    seller_id:  int
    symbol:     str
    price:      float      
    quantity:   int         
    timestamp:  int         


@dataclass
class Stock:
    symbol:            str
    price:             float          
    fundamental_value: float          
    volatility:        float          
    price_history:     list = field(default_factory=list)   
    step:              int  = 0       