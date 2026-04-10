import yaml
import random
import models
from models import Stock


def vary(value: float, pct: float = 0.10) -> float:
    return value * random.uniform(1 - pct, 1 + pct)


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
                        NoiseAgent, MarketMakerAgent,
                        MomentumAgent, MeanReversionAgent,
                        HerdAgent, ArbitrageAgent)

    agents   = []
    agent_id = 0
    cfg      = config["agents"]
    cash     = config["simulation"]["starting_cash"]
    stocks   = [s["symbol"] for s in config["stocks"]]

    def starting_holdings(cash: float, symbols: list) -> dict:
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
            initial_cash   = cash * 0.5,
            buy_threshold  = vary(f["buy_threshold"]),
            sell_threshold = vary(f["sell_threshold"]),
            risk_aversion  = vary(f["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)  # ← half as shares
        agents.append(agent)
        agent_id += 1

    c = cfg["chartist"]
    for _ in range(c["count"]):
        agent = ChartistAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(2, int(vary(c["lookback"], pct=0.30))),
            risk_aversion = vary(c["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    n = cfg["noise"]
    for _ in range(n["count"]):
        agent = NoiseAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            hold_prob     = min(vary(n["hold_prob"]), 0.95),
            buy_prob      = vary(n["buy_prob"]),
            risk_aversion = vary(n["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    m = cfg["market_maker"]
    for _ in range(m["count"]):
        agent = MarketMakerAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            spread_pct    = vary(m["spread_pct"]),
            risk_aversion = vary(m["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    mo = cfg["momentum"]
    for _ in range(mo["count"]):
        agent = MomentumAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(2, int(vary(mo["lookback"], pct=0.30))),
            threshold     = vary(mo["threshold"]),
            risk_aversion = vary(mo["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    mr = cfg["mean_reversion"]
    for _ in range(mr["count"]):
        agent = MeanReversionAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(5, int(vary(mr["lookback"], pct=0.30))),
            entry_z       = vary(mr["entry_z"]),
            risk_aversion = vary(mr["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    h = cfg["herd"]
    for _ in range(h["count"]):
        agent = HerdAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            sensitivity   = vary(h["sensitivity"]),
            risk_aversion = vary(h["risk_aversion"]),
        )
        agent.holdings = starting_holdings(cash, stocks)
        agents.append(agent)
        agent_id += 1

    ar = cfg["arbitrage"]
    for _ in range(ar["count"]):
        agent = ArbitrageAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(10, int(vary(ar.get("lookback", 30), pct=0.30))),
            threshold     = vary(ar.get("threshold", 1.5)),
            risk_aversion = vary(ar["risk_aversion"]),
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