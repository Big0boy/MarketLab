from __future__ import annotations
import random
import copy
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation

from agents import (FundamentalistAgent, ChartistAgent,
                    NoiseAgent, MarketMakerAgent)


def gini(wealths: list[float]) -> float:
    wealths = np.array(wealths, dtype=float)
    n = len(wealths)
    if n == 0 or np.mean(wealths) == 0:
        return 0.0
    mean_wealth = np.mean(wealths)
    diff_sum = np.sum(np.abs(wealths[:, None] - wealths[None, :]))
    return diff_sum / (2 * n ** 2 * mean_wealth)

def fitness_score(agents: list, exchange, initial_wealth: float) -> None:
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


def _starting_holdings(config: dict, starting_cash: float) -> dict:
    stocks = config["stocks"]
    per_stock_value = (starting_cash * 0.5) / len(stocks)
    return {
        s["symbol"]: int(per_stock_value / s["start_price"])
        for s in stocks
    }

def select_and_reproduce(sim: Simulation) -> list:
    config        = sim.config
    starting_cash = config["simulation"]["starting_cash"]

    fitness_score(sim.agents, sim.exchange, starting_cash)
    from collections import defaultdict
    by_type: dict = defaultdict(list)
    for agent in sim.agents:
        by_type[type(agent)].append(agent)
 
    def reset(agent) -> None:
        agent.cash           = starting_cash * 0.5
        agent.holdings       = _starting_holdings(config, starting_cash)
        agent.wealth_history = []
        agent.trade_history  = []
        agent.fitness        = 0.0
        agent._raw_fitness   = 0.0
 
    new_agents: list = []
 
    for agent_type, group in by_type.items():
        ranked  = sorted(group, key=lambda a: a.fitness, reverse=True)
        n       = len(ranked)
        n_elite = max(1, int(round(n * 0.20)))
        n_dead  = max(1, int(round(n * 0.20)))
 
        elite  = ranked[:n_elite]
        middle = ranked[n_elite: n - n_dead]
        dead   = ranked[n - n_dead:]
 
        for agent in elite + middle:
            reset(agent)
            mutate(agent)
            new_agents.append(agent)
 
        for dead_agent in dead:
            parent = random.choice(elite)
            child  = copy.deepcopy(parent)
            child.agent_id = dead_agent.agent_id
            reset(child)
            mutate(child)
            new_agents.append(child)
 
    new_agents.sort(key=lambda a: a.agent_id)
    return new_agents

def perform_evo(sim: Simulation) -> list:
    evolved = select_and_reproduce(sim)
    print(f"  [evo] generation complete — {len(evolved)} agents carried forward")
    return evolved