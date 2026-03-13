from simulation import Simulation


def save_output(sim):
    results  = sim.get_results()
    rankings = sim.get_agent_rankings()
    seed     = sim.config["simulation"]["random_seed"]

    with open("outputs/output.txt", "w") as f:
        f.write("  MarketLab — Simulation Results\n")
        f.write(f"  Seed:         {seed}\n")
        f.write(f"  Steps:        {results['total_steps']}\n")
        f.write(f"  Total trades: {results['total_trades']}\n")
        f.write(f"  Survivors:    {len(results['survivors'])}\n")
        f.write(f"  Bankruptcies: {len(results['bankruptcies'])}\n")
        f.write(f"  Winner:       Agent {results['winner']}\n")

        f.write("\n Final Prices\n")
        for symbol, price in results["final_prices"].items():
            f.write(f"  {symbol:8} ₹{price:.2f}\n")

        f.write("\n Agent Rankings\n")
        for i, agent in enumerate(rankings, 1):
            f.write(f"  {i:3}. Agent {agent['agent_id']:3} "
                    f"({agent['agent_type']:15}) "
                    f"₹{agent['wealth']:,.2f}\n")

        if results["bankruptcies"]:
            f.write("\n Bankruptcies\n")
            f.write("-" * 50 + "\n")
            for agent_id in results["bankruptcies"]:
                f.write(f"  Agent {agent_id}\n")

        f.write("\n trade_log\n")
        f.write(f"{sim.get_trade_log()}\n")


    print("Results saved to output.txt ✅")


def main():
    sim = Simulation("config.yaml")
    print("  MarketLab — Starting Simulation")
    print(f"  Agents:  {len(sim.agents)}")
    print(f"  Stocks:  {list(sim.exchange.order_book.stocks.keys())}")
    print(f"  Steps:   {sim.num_steps}")

    sim.run()
    save_output(sim)      # ← call here


if __name__ == "__main__":
    main()