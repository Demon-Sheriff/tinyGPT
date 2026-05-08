"""
Analysis for the D/V scaling law pilot.

Reads result.json from each run, fits scaling laws, produces clean plots.
No em dashes, soft palette, math labels.
"""
import os
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Plot style. Clean and paper-ready.
rcParams.update({
    "figure.figsize": (8, 5.5),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
})

# Colorblind-friendly palette.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00",
          "#F0E442", "#7C7C7C", "#000000"]


def load_results(out_root: str):
    """Load all result.json files under out_root."""
    rows = []
    for path in glob.glob(os.path.join(out_root, "*", "result.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def power_law(x, a, alpha, c):
    """L(x) = a * x^(-alpha) + c"""
    return a * np.power(x, -alpha) + c


def fit_power_law(x, y):
    """Fit y = a * x^(-alpha) + c. Returns (a, alpha, c)."""
    p0 = [1.0, 0.5, np.min(y) * 0.9]
    try:
        popt, _ = curve_fit(power_law, x, y, p0=p0, maxfev=10000)
        return popt
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def plot_bpb_vs_vd_ratio(rows, output_path):
    """Plot BPB vs V/D, one line per D, points colored by V."""
    fig, ax = plt.subplots()

    Ds = sorted(set(r["D"] for r in rows))
    for i, D in enumerate(Ds):
        sub = sorted([r for r in rows if r["D"] == D], key=lambda r: r["v_over_d"])
        xs = [r["v_over_d"] for r in sub]
        ys = [r["final_val_bpb"] for r in sub]
        ax.plot(xs, ys, "o-", color=COLORS[i], label=f"D = {D}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$V/D$ ratio")
    ax.set_ylabel("Validation bits per byte (lower is better)")
    ax.set_title("BPB vs V/D ratio at fixed compute (1e17 FLOPs)")
    ax.legend(title="Hidden dim")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_bpb_vs_v_per_d(rows, output_path):
    """Plot BPB vs V, one line per D."""
    fig, ax = plt.subplots()

    Ds = sorted(set(r["D"] for r in rows))
    for i, D in enumerate(Ds):
        sub = sorted([r for r in rows if r["D"] == D], key=lambda r: r["V"])
        xs = [r["V"] for r in sub]
        ys = [r["final_val_bpb"] for r in sub]
        ax.plot(xs, ys, "o-", color=COLORS[i], label=f"D = {D}")

    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary size V")
    ax.set_ylabel("Validation bits per byte")
    ax.set_title("BPB vs vocabulary size, by hidden dim")
    ax.legend(title="Hidden dim")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_isoflop_curve(rows, output_path):
    """All points on one chart, x = V, y = BPB, color = D."""
    fig, ax = plt.subplots()
    Ds = sorted(set(r["D"] for r in rows))
    for i, D in enumerate(Ds):
        sub = [r for r in rows if r["D"] == D]
        sub = sorted(sub, key=lambda r: r["V"])
        ax.plot([r["V"] for r in sub], [r["final_val_bpb"] for r in sub],
                "o-", color=COLORS[i], label=f"D = {D}")
    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary size V")
    ax.set_ylabel("Final validation BPB")
    ax.set_title("IsoFLOP slice: BPB vs V at 1e17 FLOPs")
    ax.legend(title="Hidden dim")
    fig.savefig(output_path)
    plt.close(fig)


def fit_joint(rows, flops_def="total"):
    """Fit BPB(V, D) = A * (V/D)^delta + B + e and report fit quality."""
    x = np.array([r["v_over_d"] for r in rows])
    y = np.array([r["final_val_bpb"] for r in rows])

    def model(x, a, delta, c):
        return a * np.power(x, delta) + c

    try:
        popt, pcov = curve_fit(model, x, y, p0=[0.1, 0.5, np.min(y)],
                               maxfev=10000)
        a, delta, c = popt
        y_pred = model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"Joint fit: BPB = {a:.4f} * (V/D)^{delta:.3f} + {c:.4f}")
        print(f"  R^2 = {r2:.4f}")
        return popt, r2
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default="out")
    p.add_argument("--plot_dir", type=str, default="plots")
    args = p.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    rows = load_results(args.out_root)
    if not rows:
        print(f"No results found in {args.out_root}/*/result.json")
        return

    print(f"Loaded {len(rows)} runs.")
    for r in sorted(rows, key=lambda r: (r["D"], r["V"])):
        print(f"  V={r['V']:>5} D={r['D']:>5} V/D={r['v_over_d']:>6.2f} "
              f"BPB={r['final_val_bpb']:.4f} N={r['n_total_params']/1e6:.1f}M")

    plot_bpb_vs_vd_ratio(rows, os.path.join(args.plot_dir, "bpb_vs_vd_ratio.png"))
    plot_bpb_vs_v_per_d(rows, os.path.join(args.plot_dir, "bpb_vs_V.png"))
    plot_isoflop_curve(rows, os.path.join(args.plot_dir, "isoflop_curve.png"))

    print("\nFit BPB = a * (V/D)^delta + c:")
    fit_joint(rows)


if __name__ == "__main__":
    main()
