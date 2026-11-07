"""Draw the paper's six-panel results figure from the files written by run.py.

    python plot.py [--results results]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"CLCB-SC": "tab:blue", "CUCB-SC": "tab:orange", "CLCB-SC-LS": "tab:green",
          "Epsilon-Greedy": "tab:red", "LFU": "tab:purple"}
MARKERS = {"CLCB-SC": "o", "CUCB-SC": "v", "CLCB-SC-LS": "x", "Epsilon-Greedy": "s", "LFU": "^"}
HATCHES = {"CLCB-SC": "/", "CUCB-SC": "\\", "CLCB-SC-LS": "|", "Epsilon-Greedy": "-", "LFU": "x"}


def mean_std(values, axis=-1):
    values = np.asarray(values)
    return values.mean(axis=axis), values.std(axis=axis)


def optimal_panel(ax, res):
    x = np.arange(len(res["cache_sizes"])) * 2
    styles = [("Brute Force", "/", "tab:blue"), ("Reverse Greedy", "\\", "tab:orange"), ("LFU", "x", "tab:purple")]
    for (label, hatch, color), dx in zip(styles, (-0.5, 0, 0.5)):
        ax.bar(x + dx, res[label], width=0.5, label=label, hatch=hatch, edgecolor="black", facecolor=color)
    ax.set_xticks(x, [str(k) for k in res["cache_sizes"]])
    ax.set_xlabel(r"Cache Size $(k)$")
    ax.set_ylabel("Optimal Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(r"(a) Optimal Loss vs $k$", fontsize=16)


def offline_panel(ax, res):
    for name, gaps in res["gaps"].items():
        mean, std = mean_std(gaps)
        ax.plot(res["lengths"], mean, marker=MARKERS[name], label=name, color=COLORS[name])
        ax.fill_between(res["lengths"], mean - std, mean + std, alpha=0.2, color=COLORS[name])
    ax.set_xlabel("Stream Length")
    ax.set_ylabel("Final Suboptimality Gap")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.set_title("(b) Final Suboptimality Gap vs Stream Length", fontsize=16)


def online_panel(ax, res):
    for name, regret in res["average_regret"].items():
        mean, std = mean_std(regret, axis=0)
        rounds = np.arange(len(mean))
        ax.plot(rounds, mean, marker=MARKERS[name], markevery=25, markersize=6, label=name, color=COLORS[name])
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.2, color=COLORS[name])
    ax.set_xlabel(r"Rounds $(t)$")
    ax.set_ylabel("Average Regret")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.set_title(r"(c) Avg Regret vs $t$", fontsize=16)


def bar_panel(ax, res, xlabel, title):
    x = np.arange(len(res["values"]))
    for i, (name, gaps) in enumerate(res["final_gap"].items()):
        mean, std = mean_std(gaps)
        ax.bar(x + (i - 2) * 0.15, mean, 0.15, yerr=std, capsize=4, label=name,
               facecolor=COLORS[name], edgecolor="black", hatch=HATCHES[name])
    ax.set_xticks(x, [str(v) for v in res["values"]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final Avg Regret")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=16)


def table_panel(ax, res):
    rows = [[name, f"{np.mean(res['switches'][name]):.1f}", f"{np.mean(res['runtime'][name]):.3g}"]
            for name in res["switches"] if name != "LFU"]
    table = ax.table(cellText=rows, colLabels=["Algorithm", "Mean Switches", "Mean Runtime (s)"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2)
    for (row, _), cell in table.get_celld().items():
        cell.visible_edges = "TB" if row == 0 else ("B" if row == len(rows) else "")
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.axis("off")
    ax.set_title("(f) Cache-Switch and Runtime Stats", fontsize=16)


def make_figure(results_dir):
    results_dir = Path(results_dir)
    res = {name: json.loads((results_dir / f"{name}.json").read_text())
           for name in ("optimal", "offline", "online", "cache_size", "num_queries")}
    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "axes.titlepad": 4, "axes.labelpad": 1})
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    optimal_panel(axes[0], res["optimal"])
    offline_panel(axes[1], res["offline"])
    online_panel(axes[2], res["online"])
    bar_panel(axes[3], res["cache_size"], r"Cache Size $(k)$", r"(d) Final Avg Regret vs $k$")
    bar_panel(axes[4], res["num_queries"], r"Number of Unique Queries $(m)$", r"(e) Final Avg Regret vs $m$")
    table_panel(axes[5], res["online"])
    for ax in axes[:5]:
        ax.legend()
    fig.tight_layout()
    path = results_dir / "figure.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    print(f"Saved {make_figure(parser.parse_args().results)}")
