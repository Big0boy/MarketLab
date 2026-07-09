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


MIN_ORDER_SIZE: int = 5   

@dataclass
class Order:
    agent_id:   int
    symbol:     str
    order_type: OrderType
    price:      float
    quantity:   int
    timestamp:  int
    ttl:        int = 10    


@dataclass
class Trade:
    buyer_id:   int
    seller_id:  int
    symbol:     str
    price:      float      
    quantity:   int         
    timestamp:  int         


@dataclass
class EarningsEvent:
    """Fired each time a stock has an earnings announcement."""
    symbol:            str
    step:              int
    true_surprise:     float   # actual % change to fundamental_value
    public_signal:     float   # noisy signal broadcast to all agents
    dividend_per_share: float = 0.0  # cash paid per share held on earnings day


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
    # Change 1: per-step cost of holding a short position (varies per stock)
    borrow_rate:       float = 0.0001
    # Change 3: number of remaining steps this stock is halted from trading
    halted_steps:      int = 0
    # Change 5: limited float
    total_shares:           int = 1_000_000
    shares_in_circulation:  int = 1_000_000
    # Feature 3: minimum price increment (tick size)
    tick_size:         float = 0.05
    # Feature 2: rolling average daily volume for sqrt-volume slippage
    adv:               float = 1000.0
    # Earnings: schedule and history
    earnings_interval:    int   = 90      # steps between announcements
    earnings_surprise_vol: float = 0.06   # σ of fundamental jump per announcement
    next_earnings:        int   = 90      # step of next announcement
    earnings_history:     list  = field(default_factory=list)  # list[EarningsEvent]
    # Dividend yield paid to long holders on each earnings day (as fraction of price)
    dividend_yield:       float = 0.005   # 0.5% of price per earnings announcement