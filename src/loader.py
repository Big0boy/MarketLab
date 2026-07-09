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
        total_shares = int(vary(s.get("total_shares", 1_000_000), pct=0.05))
        stock = Stock(
            symbol            = s["symbol"],
            price             = s["start_price"],
            fundamental_value = s["fundamental_value"],
            initial_fundamental = s["fundamental_value"],
            volatility        = s["volatility"],
            borrow_rate       = s.get("borrow_rate", 0.0001),
            total_shares      = total_shares,
            shares_in_circulation = total_shares,
            tick_size         = s.get("tick_size", 0.05),
            adv               = s.get("adv", 1000.0),
            earnings_interval    = s.get("earnings_interval", 90),
            earnings_surprise_vol = s.get("earnings_surprise_vol", 0.06),
            next_earnings        = s.get("earnings_interval", 90),
        )
        # seed price history so chartists have data immediately
        stock.price_history = [s["start_price"]] * 10   # ← ADD THIS
        stocks.append(stock)
    return stocks


def build_agents(config: dict, stocks: list = None) -> list:
    from agents import (FundamentalistAgent, ChartistAgent,
                        NoiseAgent, MarketMakerAgent,
                        MomentumAgent, MeanReversionAgent,
                        HerdAgent, ArbitrageAgent)

    agents   = []
    agent_id = 0
    cfg      = config["agents"]
    cash     = config["simulation"]["starting_cash"]
    stocks_cfg = config["stocks"]
    symbols  = [s["symbol"] for s in stocks_cfg]

    # Change 5: total agent count, used to check starting holdings vs float
    total_agent_count = sum(cfg[k]["count"] for k in
                             ("fundamentalist", "chartist", "noise",
                              "market_maker", "momentum", "mean_reversion",
                              "herd", "arbitrage"))

    # Map symbol -> total_shares (post-variation if `stocks` objects passed,
    # else from config)
    float_by_symbol = {}
    if stocks:
        float_by_symbol = {s.symbol: s.total_shares for s in stocks}
    else:
        for s in stocks_cfg:
            float_by_symbol[s["symbol"]] = s.get("total_shares", 1_000_000)

    def starting_holdings(cash: float, symbols: list) -> dict:
        per_stock_value = (cash * 0.5) / len(symbols)
        holdings = {}
        for s in stocks_cfg:
            price = s["start_price"]
            qty   = int(per_stock_value / price)

            # Change 5: don't let aggregate starting holdings exceed the float
            symbol      = s["symbol"]
            total_float = float_by_symbol.get(symbol, 1_000_000)
            max_per_agent = total_float // max(1, total_agent_count)
            holdings[symbol] = min(qty, max_per_agent)
        return holdings

    f = cfg["fundamentalist"]
    for _ in range(f["count"]):
        agent = FundamentalistAgent(
            agent_id          = agent_id,
            initial_cash      = cash * 0.5,
            buy_threshold     = vary(f["buy_threshold"]),
            sell_threshold    = vary(f["sell_threshold"]),
            risk_aversion     = vary(f["risk_aversion"]),
            cash_preference   = vary(f.get("cash_preference", 0.85)),
            max_ownership_pct = vary(f.get("max_ownership_pct", 0.30)),
            # Heterogeneous beliefs — each agent gets different noise/learning
            belief_noise        = vary(f.get("belief_noise", 0.05), pct=0.50),
            learning_rate       = vary(f.get("learning_rate", 0.30), pct=0.40),
            signal_sensitivity  = vary(f.get("signal_sensitivity", 1.0), pct=0.30),
        )
        agent.holdings = starting_holdings(cash, symbols)
        agents.append(agent)
        agent_id += 1

    c = cfg["chartist"]
    for _ in range(c["count"]):
        agent = ChartistAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(2, int(vary(c["lookback"], pct=0.30))),
            risk_aversion = vary(c["risk_aversion"]),
            cash_preference   = vary(c.get("cash_preference", 0.80)),
            max_ownership_pct = vary(c.get("max_ownership_pct", 0.22)),
        )
        agent.holdings = starting_holdings(cash, symbols)
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
            cash_preference   = vary(n.get("cash_preference", 0.75)),
            max_ownership_pct = vary(n.get("max_ownership_pct", 0.15)),
        )
        agent.holdings = starting_holdings(cash, symbols)
        agents.append(agent)
        agent_id += 1

    m = cfg["market_maker"]
    for _ in range(m["count"]):
        agent = MarketMakerAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            spread_pct    = vary(m["spread_pct"]),
            risk_aversion = vary(m["risk_aversion"]),
            target_inventory  = m.get("target_inventory", 0),
            cash_preference   = vary(m.get("cash_preference", 0.60)),
            max_ownership_pct = vary(m.get("max_ownership_pct", 0.15)),
        )
        agent.holdings = starting_holdings(cash, symbols)
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
            cash_preference   = vary(mo.get("cash_preference", 0.80)),
            max_ownership_pct = vary(mo.get("max_ownership_pct", 0.22)),
        )
        agent.holdings = starting_holdings(cash, symbols)
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
            cash_preference   = vary(mr.get("cash_preference", 0.80)),
            max_ownership_pct = vary(mr.get("max_ownership_pct", 0.22)),
        )
        agent.holdings = starting_holdings(cash, symbols)
        agents.append(agent)
        agent_id += 1

    h = cfg["herd"]
    for _ in range(h["count"]):
        agent = HerdAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            sensitivity   = vary(h["sensitivity"]),
            risk_aversion = vary(h["risk_aversion"]),
            cash_preference   = vary(h.get("cash_preference", 0.85)),
            max_ownership_pct = vary(h.get("max_ownership_pct", 0.15)),
        )
        agent.holdings = starting_holdings(cash, symbols)
        agents.append(agent)
        agent_id += 1

    ar = cfg["arbitrage"]
    for _ in range(ar["count"]):
        agent = ArbitrageAgent(
            agent_id      = agent_id,
            initial_cash  = cash * 0.5,
            lookback      = max(10, int(vary(ar.get("lookback", 90), pct=0.30))),
            threshold     = vary(ar.get("threshold", 1.5)),
            risk_aversion = vary(ar["risk_aversion"]),
            cash_preference   = vary(ar.get("cash_preference", 0.80)),
            max_ownership_pct = vary(ar.get("max_ownership_pct", 0.22)),
        )
        agent.holdings = starting_holdings(cash, symbols)
        agents.append(agent)
        agent_id += 1

    return agents


def set_seed(config: dict | None) -> None:
    if config is None:
        return
    seed = config["simulation"]["random_seed"]
    print(seed)
    random.seed(seed)