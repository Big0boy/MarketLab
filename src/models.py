from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    BUY  = "BUY"
    SELL = "SELL"

class AgentType(Enum):
    Base = "base"
    FUNDAMENTALIST = "fundamentalist"
    CHARTIST = "chartist"
    NOISE = "noise"
    MARKET_MAKER = "market_maker"
    Q_LEARNER = "q_learner"      
    MOMENTUM = "momentum"        
    MEAN_REVERSION = "mean_reversion"
    HERD = "herd"
    ARBITRAGE = "arbitrage" 


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
    initial_fundamental: float         
    volatility:        float          
    price_history:     list = field(default_factory=list)   
    sediment:          float = 0.0
    step:              int  = 0       