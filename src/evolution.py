from __future__ import annotations
import random
import copy
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from agents import (FundamentalistAgent, ChartistAgent,
                    NoiseAgent, MarketMakerAgent,
                    MomentumAgent, MeanReversionAgent,
                    HerdAgent, ArbitrageAgent)


# ── Gini coefficient ─────────────────────────────────────────────────────────

def gini(wealths: list[float]) -> float:
    wealths = np.array(wealths, dtype=float)
    n = len(wealths)
    if n == 0 or np.mean(wealths) == 0:
        return 0.0
    mean_wealth = np.mean(wealths)
    diff_sum = np.sum(np.abs(wealths[:, None] - wealths[None, :]))
    return diff_sum / (2 * n ** 2 * mean_wealth)


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness_score(agents: list, exchange, initial_wealth: float) -> None:
    """Compute and normalise fitness in-place on each agent."""
    fitness_arr = []

    for agent in agents:
        current      = agent.get_total_wealth(exchange)
        total_return = (current - initial_wealth) / max(initial_wealth, 1.0)

        wh = agent.wealth_history
        if len(wh) > 2:
            rets   = np.diff(wh) / np.maximum(np.array(wh[:-1]), 1.0)
            avg_r  = float(np.mean(rets))
            std_r  = float(np.std(rets))
            sharpe = avg_r / std_r if std_r > 1e-10 else 0.0
        else:
            sharpe = 0.0

        agent._raw_fitness = total_return * (1.0 + sharpe)
        fitness_arr.append(agent._raw_fitness)

    f_min = min(fitness_arr)
    f_max = max(fitness_arr)
    eps   = 1e-10

    for agent in agents:
        agent.fitness = (agent._raw_fitness - f_min) / (f_max - f_min + eps)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(agent, magnitude: float = 0.05) -> None:
    """Perturb an agent's hyperparameters in-place."""
    def vary(v: float) -> float:
        return v * random.uniform(1 - magnitude, 1 + magnitude)

    if isinstance(agent, FundamentalistAgent):
        agent.buy_threshold  = max(0.80, min(vary(agent.buy_threshold),  0.99))
        agent.sell_threshold = max(1.01, min(vary(agent.sell_threshold), 1.20))
        agent.risk_aversion  = max(0.1,  min(vary(agent.risk_aversion),  5.0))

    elif isinstance(agent, ChartistAgent):
        agent.lookback      = max(2, min(int(vary(agent.lookback)), 30))
        agent.risk_aversion = max(0.1, min(vary(agent.risk_aversion), 5.0))

    elif isinstance(agent, NoiseAgent):
        agent.hold_prob     = max(0.05, min(vary(agent.hold_prob), 0.95))
        agent.buy_prob      = max(0.05, min(vary(agent.buy_prob),  0.90))
        agent.risk_aversion = max(0.1,  min(vary(agent.risk_aversion), 5.0))

    elif isinstance(agent, MarketMakerAgent):
        agent.spread_pct    = max(0.001, min(vary(agent.spread_pct),    0.05))
        agent.risk_aversion = max(0.1,   min(vary(agent.risk_aversion), 5.0))

    elif isinstance(agent, MomentumAgent):
        agent.lookback      = max(2,    min(int(vary(agent.lookback)), 50))
        agent.threshold     = max(0.001, min(vary(agent.threshold),    0.05))
        agent.risk_aversion = max(0.1,   min(vary(agent.risk_aversion), 5.0))

    elif isinstance(agent, MeanReversionAgent):
        agent.lookback      = max(5,    min(int(vary(agent.lookback)), 60))
        agent.entry_z       = max(0.001, min(vary(agent.entry_z),      0.10))
        agent.risk_aversion = max(0.1,   min(vary(agent.risk_aversion), 5.0))

    elif isinstance(agent, HerdAgent):
        agent.sensitivity   = max(0.1,  min(vary(agent.sensitivity),   5.0))
        agent.risk_aversion = max(0.1,  min(vary(agent.risk_aversion),  5.0))

    elif isinstance(agent, ArbitrageAgent):
        agent._ratio_history = {}
        agent.lookback      = max(10,   min(int(vary(agent.lookback)),   60))
        agent.threshold     = max(0.5,  min(vary(agent.threshold),       4.0))
        agent.risk_aversion = max(0.1,  min(vary(agent.risk_aversion),   5.0))


# ── Starting holdings (mirrors loader.build_agents logic) ────────────────────

def _starting_holdings(config: dict, starting_cash: float) -> dict:
    """
    Mirrors loader.build_agents:
      half of starting_cash as cash (already set on agent),
      the other half spread evenly across stocks as shares.
    """
    stocks = config["stocks"]
    per_stock_value = (starting_cash * 0.5) / len(stocks)
    return {
        s["symbol"]: int(per_stock_value / s["start_price"])
        for s in stocks
    }


# ── Selection & reproduction ──────────────────────────────────────────────────

def select_and_reproduce(sim: Simulation) -> list:
    from collections import defaultdict

    config        = sim.config
    starting_cash = config["simulation"]["starting_cash"]

    fitness_score(sim.agents, sim.exchange, starting_cash)

    # Map config type names → class and configured count
    type_cfg = {
        FundamentalistAgent: config["agents"]["fundamentalist"]["count"],
        ChartistAgent:       config["agents"]["chartist"]["count"],
        NoiseAgent:          config["agents"]["noise"]["count"],
        MarketMakerAgent:    config["agents"]["market_maker"]["count"],
        MomentumAgent:       config["agents"]["momentum"]["count"],
        MeanReversionAgent:  config["agents"]["mean_reversion"]["count"],
        HerdAgent:           config["agents"]["herd"]["count"],
        ArbitrageAgent:      config["agents"]["arbitrage"]["count"],
    }

    original_ids_by_type: dict = defaultdict(list)
    for agent_type, target_n in type_cfg.items():
        pass  # filled below from the full original id range

    # Reconstruct original id ranges from config counts in insertion order
    agent_id = 0
    for agent_type in [FundamentalistAgent, ChartistAgent, NoiseAgent, MarketMakerAgent,
                       MomentumAgent, MeanReversionAgent, HerdAgent, ArbitrageAgent]:
        target_n = type_cfg[agent_type]
        original_ids_by_type[agent_type] = list(range(agent_id, agent_id + target_n))
        agent_id += target_n

    # Group surviving agents by type
    survivors_by_type: dict = defaultdict(list)
    for agent in sim.agents:
        survivors_by_type[type(agent)].append(agent)

    def reset(agent) -> None:
        agent.cash           = starting_cash * 0.5
        agent.holdings       = _starting_holdings(config, starting_cash)
        agent.wealth_history = []
        agent.trade_history  = []
        agent.fitness        = 0.0
        agent._raw_fitness   = 0.0

    new_agents: list = []

    for agent_type, target_n in type_cfg.items():
        survivors = survivors_by_type.get(agent_type, [])
        ranked    = sorted(survivors, key=lambda a: a.fitness, reverse=True)

        if ranked:
            n       = len(ranked)
            n_elite = max(1, int(round(n * 0.20)))
            n_dead  = max(1, int(round(n * 0.20)))

            elite  = ranked[:n_elite]
            middle = ranked[n_elite: n - n_dead]
            dead   = ranked[n - n_dead:]
        else:
            # Entire type wiped out — will be fully restored below
            elite = middle = dead = []

        # Keep survivors (elite + middle), replace dead with offspring
        kept: list = []
        for agent in elite + middle:
            reset(agent)
            mutate(agent)
            kept.append(agent)

        for dead_agent in dead:
            if elite:
                parent = random.choice(elite)
                child  = copy.deepcopy(parent)
            else:
                # No survivors at all for this type — create a fresh one
                child = agent_type(agent_id=dead_agent.agent_id,
                                   initial_cash=starting_cash * 0.5)
            child.agent_id = dead_agent.agent_id
            reset(child)
            mutate(child)
            kept.append(child)

        # Restore any slots lost to bankruptcy — fill up to target_n
        current_ids  = {a.agent_id for a in kept}
        all_ids      = original_ids_by_type[agent_type]
        missing_ids  = [i for i in all_ids if i not in current_ids]

        for missing_id in missing_ids:
            if elite:
                parent = random.choice(elite)
                child  = copy.deepcopy(parent)
            else:
                child = agent_type(agent_id=missing_id,
                                   initial_cash=starting_cash * 0.5)
            child.agent_id = missing_id
            reset(child)
            mutate(child)
            kept.append(child)

        new_agents.extend(kept)

    new_agents.sort(key=lambda a: a.agent_id)
    return new_agents


# ── Main hook called from main.py ─────────────────────────────────────────────

def perform_evo(sim: Simulation) -> list:
    """
    Called by main.py after a simulation run completes.
    Returns the evolved agent list; main.py is responsible for
    injecting it into the next Simulation instance.
    """
    evolved = select_and_reproduce(sim)
    print(f"  [evo] generation complete — {len(evolved)} agents carried forward")
    return evolved