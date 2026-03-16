import random

from exchange import Exchange
from loader import load_config, build_stocks, build_agents
import models
import evolution as evo


class Simulation:

    def __init__(self, config_path: str = "config.yaml", agents: list | None = None):
        self.config = load_config(config_path)
        seed = self.config["simulation"]["random_seed"]
        random.seed(seed)
        print(f"  Seed: {seed}")
        self.step             = 0
        self.log              = []
        self.all_bankrupt_ids = set()   
        self.bankrupt_agents  = []    
        self.stocks           = build_stocks(self.config)
        self.agents           = agents if agents is not None else build_agents(self.config)
        self.exchange         = Exchange(self.stocks)
        self.num_steps        = self.config["simulation"]["steps"]

        for agent in self.agents:
            self.exchange.register_agent(agent)


    def run(self) -> None:
        for _ in range(self.num_steps):
            self._step()

    def _step(self) -> None:
        self.bankrupt_agents = []
        for agent in list(self.agents):       
            if agent.is_bankrupt(self.exchange):
                self.log.append(
                    f"Agent {agent.agent_id} went bankrupt at step {self.step}."
                )
                self.exchange.agents.pop(agent.agent_id, None)
                self.all_bankrupt_ids.add(agent.agent_id)
                self.bankrupt_agents.append(agent)
                self.agents.remove(agent)
                continue

            order = agent.decide(self.exchange)
            if order:
                self.exchange.submit_order(order)
            agent.record_wealth(self.exchange)

        trades = self.exchange.run_step()
        self._log_step(trades)
        self.step += 1

    def _log_step(self, trades: list[models.Trade]) -> None:
        self.log.append({
            "step":         self.step,
            "prices":       {sym: stock.price
                             for sym, stock in self.exchange.order_book.stocks.items()},
            "trade_count":  len(trades),
            "bankruptcies": {a.agent_id for a in self.bankrupt_agents},
            "top_agents":   sorted(
                                self.agents,
                                key=lambda a: a.get_total_wealth(self.exchange),
                                reverse=True,
                            )[:5],
        })


    def _wealth_list(self) -> list[float]:
        return [agent.get_total_wealth(self.exchange) for agent in self.agents]

    def get_results(self) -> dict:
        winner = (
            max(self.agents, key=lambda a: a.get_total_wealth(self.exchange))
            if self.agents else None
        )
        return {
            "total_steps":  self.num_steps,
            "total_trades": len(self.exchange.trade_log),
            "survivors":    {a.agent_id for a in self.agents
                             if not a.is_bankrupt(self.exchange)},
            "bankruptcies": self.all_bankrupt_ids,
            "winner":       winner.agent_id if winner else None,
            "final_prices": {sym: stock.price
                             for sym, stock in self.exchange.order_book.stocks.items()},
            "gini":         evo.gini(self._wealth_list()),
        }

    def get_agent_rankings(self) -> list[dict]:
        evo.fitness_score(
            self.agents,
            self.exchange,
            self.config["simulation"]["starting_cash"],
        )
        return sorted(
            [
                {
                    "agent_id":   agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "wealth":     agent.get_total_wealth(self.exchange),
                    "fitness":    agent.fitness,
                }
                for agent in self.agents
            ],
            key=lambda x: x["fitness"],
            reverse=True,
        )

    def get_trade_log(self) -> list[models.Trade]:
        return self.exchange.trade_log