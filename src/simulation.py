import random

from exchange import Exchange
from loader import load_config, build_stocks, build_agents, set_seed
import models

class Simulation():

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        seed = self.config["simulation"]["random_seed"]
        random.seed(seed)
        print(f"  Seed: {random.seed}") 
        self.step : int = 0
        self.log = []
        self.config = load_config(config_path)
        self.stocks = build_stocks(self.config)
        self.agents = build_agents(self.config)
        self.exchange = Exchange(self.stocks)
        set_seed(self.config)
        self.num_steps       = self.config["simulation"]["steps"]
        for agent in self.agents:
            self.exchange.register_agent(agent)
        
    def run(self) -> dict:
        for step in range(self.num_steps):
            self._step()
        pass
        
    def _step(self) -> None:
        self.bankrupt_agents : list[models.Agent] = []
        for agent in self.agents:
            if agent.is_bankrupt(self.exchange):
                self.log.append(f"Agent {agent.agent_id} went bankrupt at step {self.step}.")
                self.exchange.agents.pop(agent.agent_id, None)
                self.bankrupt_agents.append(agent)
                continue
            order = agent.decide(self.exchange)
            if order:
                self.exchange.submit_order(order)
            agent.record_wealth(self.exchange)
        trades = self.exchange.run_step()
        self._log_step(trades)
        self.step += 1

    def _log_step(self, trades: list[models.Trade]) -> None:
        step_log = {
            "step": self.step,
            "prices": {sym: stock.price for sym, stock in self.exchange.order_book.stocks.items()},
            "trade_count": len(trades),
            "bankruptcies": {agent.agent_id for agent in self.bankrupt_agents},
            "top_agents": sorted(self.agents, key=lambda a: a.get_total_wealth(self.exchange), reverse=True)[:5],
        }
        self.log.append(step_log)


    def get_results(self) -> dict:
        return {
           "total_steps":    self.num_steps,
           "total_trades":   len(self.exchange.trade_log),
           "survivors":      {agent.agent_id for agent in self.agents if not agent.is_bankrupt(self.exchange)},
           "bankruptcies":   {agent.agent_id for agent in self.bankrupt_agents},
           "winner":         max(self.agents, key=lambda a: a.get_total_wealth(self.exchange)).agent_id,
           "final_prices":   {sym: stock.price for sym, stock in self.exchange.order_book.stocks.items()},
           "price_history":  {sym: stock.price_history for sym, stock in self.exchange.order_book.stocks.items()}
        }
    
    def get_agent_rankings(self) -> list:
        return sorted(
            [
                {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "wealth": agent.get_total_wealth(self.exchange)
                }
                for agent in self.agents
            ],
            key=lambda x: x["wealth"],
            reverse=True
        )

    def get_trade_log(self) -> list[models.Trade]:
        return self.exchange.trade_log
    