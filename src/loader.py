import yaml
import random
import models
from models import Stock


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_stocks(config: dict) -> list[Stock]:
    stocks = []
    for s in config["stocks"]:
        stock = Stock(
            symbol            = s["symbol"],
            price             = s["start_price"],
            fundamental_value = s["fundamental_value"],
            initial_fundamental = s["fundamental_value"],
            volatility        = s["volatility"],
        )
        # seed price history so chartists have data immediately
        stock.price_history = [s["start_price"]] * 10   # ← ADD THIS
        stocks.append(stock)
    return stocks


def build_agents(config: dict) -> list:
    from agents import (FundamentalistAgent, ChartistAgent,
                        NoiseAgent, MarketMakerAgent)

    agents   = []
    agent_id = 0
    cfg      = config["agents"]
    cash     = config["simulation"]["starting_cash"]
    stocks   = [s["symbol"] for s in config["stocks"]]

    # helper — split cash: half as cash, half as holdings
    def starting_holdings(cash: float, symbols: list) -> dict:
        # give each agent shares worth ~half their cash
        # split evenly across all stocks
        per_stock_value = (cash * 0.5) / len(symbols)
        holdings = {}
        for s in config["stocks"]:
            price = s["start_price"]
            holdings[s["symbol"]] = int(per_stock_value / price)
        return holdings

    f = cfg["fundamentalist"]
    for _ in range(f["count"]):
        agent = FundamentalistAgent(
            agent_id       = agent_id,
            initial_cash   = cash * 0.5,       # ← half cash
            buy_threshold  = f["buy_threshold"],
            sell_threshold = f["sell_threshold"],
            risk_aversion  = f["risk_aversion"],
        )
        agent.holdings = starting_holdings(cash, stocks)  # ← half as shares
        agents.append(agent)
        agent_id += 1

    c = cfg["chartist"]
    for _ in range(c["count"]):
        agent = ChartistAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = c["lookback"],
            risk_aversion = c["risk_aversion"],
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    n = cfg["noise"]
    for _ in range(n["count"]):
        agent = NoiseAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            hold_prob     = n["hold_prob"],
            buy_prob      = n["buy_prob"],
            risk_aversion = n["risk_aversion"],
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    m = cfg["market_maker"]
    for _ in range(m["count"]):
        agent = MarketMakerAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            spread_pct    = m["spread_pct"],
            risk_aversion = m["risk_aversion"],
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    return agents


def set_seed(config: dict | None) -> None:
    if config is None:
        return
    seed = config["simulation"]["random_seed"]
    print(seed)
    random.seed(seed)