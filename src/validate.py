import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from simulation import Simulation


# ── 1. Run Simulation ────────────────────────────────────────────────────────

def run_simulation() -> Simulation:
    print("=" * 50)
    print("  MarketLab — Stylized Facts Validation")
    print("=" * 50)
    print("Running simulation...")
    sim = Simulation("config.yaml")
    sim.run()
    results = sim.get_results()
    print(results)
    print("=" * 50)
    return sim


# ── 2. Extract Data ──────────────────────────────────────────────────────────

def extract_data(sim: Simulation, symbol: str = "AAPL") -> dict:
    prices      = np.array(sim.exchange.order_book.stocks[symbol].price_history)
    returns     = np.diff(prices) / prices[:-1]
    returns     = returns[returns != 0]
    abs_returns = np.abs(returns)
    trade_sizes = np.array([t.quantity for t in sim.exchange.trade_log])

    print(f"\n  Symbol:      {symbol}")
    print(f"  Price steps: {len(prices)}")
    print(f"  Returns:     {len(returns)}")
    print(f"  Trades:      {len(trade_sizes)}")

    return {
        "symbol":      symbol,
        "prices":      prices,
        "returns":     returns,
        "abs_returns": abs_returns,
        "trade_sizes": trade_sizes,
    }


# ── 3. Compute Stats ─────────────────────────────────────────────────────────

def compute_stats(data: dict) -> dict:
    returns     = data["returns"]
    abs_returns = data["abs_returns"]
    trade_sizes = data["trade_sizes"]

    kurtosis   = float(stats.kurtosis(returns, fisher=False))
    skewness   = float(stats.skew(returns))
    volatility = float(np.std(returns))

    if len(abs_returns) > 2:
        autocorr = float(np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1])
    else:
        autocorr = 0.0

    size_skew = float(stats.skew(trade_sizes)) if len(trade_sizes) > 0 else 0.0

    return {
        "kurtosis":   kurtosis,
        "skewness":   skewness,
        "volatility": volatility,
        "autocorr":   autocorr,
        "size_skew":  size_skew,
    }


# ── 4. Pass / Fail Report ────────────────────────────────────────────────────

def print_report(s: dict) -> None:
    def result(condition): return "✅  PASS" if condition else "❌  FAIL"

    fat_tails   = s["kurtosis"]  > 3.0
    vol_cluster = s["autocorr"]  > 0.1
    power_law   = s["size_skew"] > 0.5

    print("\n📊  STYLIZED FACTS REPORT")
    print("=" * 50)
    print(f"  Kurtosis:           {s['kurtosis']:.3f}  (need > 3.0)")
    print(f"  Skewness:           {s['skewness']:.3f}")
    print(f"  Volatility (std):   {s['volatility']:.6f}")
    print(f"  Autocorr |returns|: {s['autocorr']:.3f}  (need > 0.1)")
    print(f"  Trade size skew:    {s['size_skew']:.3f}  (need > 0.5)")
    print("-" * 50)
    print(f"  1. Fat Tails:             {result(fat_tails)}")
    print(f"  2. Volatility Clustering: {result(vol_cluster)}")
    print(f"  3. Power Law (sizes):     {result(power_law)}")
    print("=" * 50)

    passed = sum([fat_tails, vol_cluster, power_law])
    print(f"\n  Score: {passed}/3 stylized facts confirmed")
    if passed == 3:
        print("  🎉 Market behaves like a real stock market!")
    elif passed == 2:
        print("  ⚠️  Close — tune parameters to improve")
    else:
        print("  ❌  Market needs tuning")
    print()


# ── 5. Plot Charts ───────────────────────────────────────────────────────────

BLUE   = "#58A6FF"
GREEN  = "#3FB950"
YELLOW = "#D29922"
ORANGE = "#FFA657"
MUTED  = "#8B949E"
WHITE  = "#E6EDF3"
DARK   = "#161B22"
BG     = "#0D1117"


def style_ax(ax, title: str) -> None:
    ax.set_facecolor(DARK)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")


def plot_charts(data: dict, s: dict, generation: int = 0) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gen_label = f" — Generation {generation}" if generation else ""
    fig.suptitle(f"MarketLab — Stylized Facts Validation{gen_label}",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    prices      = data["prices"]
    returns     = data["returns"]
    abs_returns = data["abs_returns"]
    trade_sizes = data["trade_sizes"]
    symbol      = data["symbol"]

    # ── Chart 1: Price History ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, f"📈  {symbol} Price History")
    ax1.plot(prices, color=BLUE, linewidth=0.8, alpha=0.9)
    ax1.fill_between(range(len(prices)), prices, prices.min(),
                     alpha=0.1, color=BLUE)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price (₹)")
    ax1.set_xlim(0, len(prices))
    ax1.annotate(f"Start: ₹{prices[0]:.2f}",
                 xy=(0, prices[0]), color=MUTED, fontsize=7,
                 xytext=(10, 5), textcoords="offset points")
    ax1.annotate(f"End: ₹{prices[-1]:.2f}",
                 xy=(len(prices)-1, prices[-1]), color=GREEN, fontsize=7,
                 xytext=(-60, 5), textcoords="offset points")

    # ── Chart 2: Fat Tails ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fat_tails_pass = s["kurtosis"] > 3.0
    style_ax(ax2, f"📊  Fat Tails  {'✅' if fat_tails_pass else '❌'}  "
                  f"(kurtosis={s['kurtosis']:.2f})")
    n_bins = min(50, max(10, len(returns) // 5))
    ax2.hist(returns, bins=n_bins, density=True,
             color=BLUE, alpha=0.7, label="Simulated returns")
    x   = np.linspace(returns.min(), returns.max(), 200)
    mu  = returns.mean()
    sig = returns.std()
    ax2.plot(x, stats.norm.pdf(x, mu, sig),
             color=ORANGE, linewidth=2, linestyle="--", label="Normal dist")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7, labelcolor=MUTED,
               facecolor=DARK, edgecolor="#30363D")

    # ── Chart 3: Volatility Clustering ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vol_pass = s["autocorr"] > 0.1
    style_ax(ax3, f"⚡  Volatility Clustering  {'✅' if vol_pass else '❌'}  "
                  f"(autocorr={s['autocorr']:.3f})")
    ax3.plot(abs_returns, color=YELLOW, linewidth=0.6, alpha=0.8)
    ax3.fill_between(range(len(abs_returns)), abs_returns,
                     alpha=0.2, color=YELLOW)
    window = max(10, len(abs_returns) // 50)
    if len(abs_returns) >= window:
        rolling = np.convolve(abs_returns, np.ones(window)/window, mode="valid")
        offset  = window // 2
        ax3.plot(range(offset, offset + len(rolling)), rolling,
                 color=ORANGE, linewidth=1.5, label=f"{window}-step avg")
        ax3.legend(fontsize=7, labelcolor=MUTED,
                   facecolor=DARK, edgecolor="#30363D")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("|Return|")
    ax3.set_xlim(0, len(abs_returns))

    # ── Chart 4: Order Size Distribution ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    power_pass = s["size_skew"] > 0.5
    style_ax(ax4, f"📦  Order Size Distribution  {'✅' if power_pass else '❌'}  "
                  f"(skew={s['size_skew']:.2f})")
    if len(trade_sizes) > 0:
        n_bins_s = min(30, len(np.unique(trade_sizes)))
        ax4.hist(trade_sizes, bins=n_bins_s, density=True,
                 color=GREEN, alpha=0.8)
        ax4.axvline(trade_sizes.mean(), color=ORANGE,
                    linestyle="--", linewidth=1.5,
                    label=f"Mean: {trade_sizes.mean():.1f}")
        ax4.set_xlabel("Order Size (shares)")
        ax4.set_ylabel("Density")
        ax4.legend(fontsize=7, labelcolor=MUTED,
                   facecolor=DARK, edgecolor="#30363D")
    else:
        ax4.text(0.5, 0.5, "No trades recorded",
                 transform=ax4.transAxes, color=MUTED,
                 ha="center", va="center")

    gen_suffix = f"_gen{generation}" if generation else ""
    filename   = f"stylized_facts_{symbol}{gen_suffix}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved → {filename}")

def main():
    sim  = run_simulation()
    data = extract_data(sim, symbol="AAPL")
    s    = compute_stats(data)
    print_report(s)
    plot_charts(data, s)


if __name__ == "__main__":
    main()