import os
from simulation import Simulation
import evolution as evo
import validate as v


CONFIG = "config.yaml"


def save_output(sim: Simulation, generation: int = 0) -> None:
    os.makedirs("outputs", exist_ok=True)
    results  = sim.get_results()
    rankings = sim.get_agent_rankings()
    seed     = sim.config["simulation"]["random_seed"]

    path = f"outputs/output_gen{generation}.txt" if generation else "outputs/output.txt"
    with open(path, "w") as f:
        f.write("  MarketLab — Simulation Results\n")
        if generation:
            f.write(f"  Generation:   {generation}\n")
        f.write(f"  Seed:         {seed}\n")
        f.write(f"  Steps:        {results['total_steps']}\n")
        f.write(f"  Total trades: {results['total_trades']}\n")
        f.write(f"  Survivors:    {len(results['survivors'])}\n")
        f.write(f"  Bankruptcies: {len(results['bankruptcies'])}\n")
        f.write(f"  Winner:       Agent {results['winner']}\n")
        f.write(f"  Gini:         {results['gini']:,.4f}\n")

        f.write("\n Final Prices\n")
        for symbol, price in results["final_prices"].items():
            f.write(f"  {symbol:8} ₹{price:.2f}\n")

        f.write("\n Agent Rankings\n")
        for i, agent in enumerate(rankings, 1):
            f.write(f"  {i:3}. Agent {agent['agent_id']:3} "
                    f"({agent['agent_type']:15}) "
                    f"₹{agent['wealth']:10.2f}"
                    f"  fitness={agent['fitness']:.4f}\n")

        if results["bankruptcies"]:
            f.write("\n Bankruptcies\n")
            f.write("-" * 50 + "\n")
            for agent_id in sorted(results["bankruptcies"]):
                f.write(f"  Agent {agent_id}\n")

    print(f"Results saved to {path} ✅")



def run_once() -> None:
    sim = Simulation(CONFIG)
    print("  MarketLab — Starting Simulation")
    print(f"  Agents:  {len(sim.agents)}")
    print(f"  Stocks:  {list(sim.exchange.order_book.stocks.keys())}")
    print(f"  Steps:   {sim.num_steps}")

    sim.run()
    save_output(sim)

    for symbol in sim.exchange.order_book.stocks.keys():
        data = v.extract_data(sim, symbol=symbol)
        s    = v.compute_stats(data)
        v.print_report(s)
        v.plot_charts(data, s)         



def run_evolution(generations: int = 5) -> None:
    evolved_agents = None  

    for gen in range(1, generations + 1):
        print(f"\n{'='*50}")
        print(f"  Generation {gen} / {generations}")
        print(f"{'='*50}")

        sim = Simulation(CONFIG, agents=evolved_agents)
        print(f"  Agents:  {len(sim.agents)}")
        print(f"  Stocks:  {list(sim.exchange.order_book.stocks.keys())}")
        print(f"  Steps:   {sim.num_steps}")

        sim.run()
        save_output(sim, generation=gen)

        for symbol in sim.exchange.order_book.stocks.keys():
            data = v.extract_data(sim, symbol=symbol)
            s    = v.compute_stats(data)
            v.print_report(s)
            v.plot_charts(data, s, generation=gen)

        evolved_agents = evo.perform_evo(sim)

    print("\n  Evolution complete ✅")



def main():
    run_once()


if __name__ == "__main__":
    main()