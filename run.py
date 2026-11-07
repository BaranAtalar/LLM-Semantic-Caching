"""Run the paper's experiments, save the results to results/<experiment>.json and draw the figure.

    python run.py                    # every experiment, then results/figure.pdf
    python run.py online cache_size  # only some experiments
"""
import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from caching import Problem, logged_dataset, offline_caches, reverse_greedy, run_online

DELTA = 0.8          # confidence parameter
K, M, T = 5, 20, 200  # default cache size, number of queries and horizon
OFFLINE_LENGTHS = [300, 325, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600]
ONLINE_ALGORITHMS = ["CLCB-SC", "CUCB-SC", "CLCB-SC-LS", "Epsilon-Greedy", "LFU"]


@lru_cache(maxsize=None)
def problem(m):
    return Problem(m)


def optimal(runs):
    """Panel (a): loss of the optimal cache found by brute force and by Reverse Greedy.

    All queries are equally likely, so LFU keeps an arbitrary set of k queries; it is drawn at random.
    """
    prob, rng = problem(M), random.Random(0)
    results = {"cache_sizes": [5, 7, 10], "Brute Force": [], "Reverse Greedy": [], "LFU": []}
    for k in results["cache_sizes"]:
        results["Brute Force"].append(prob.optimal_loss(k))
        results["Reverse Greedy"].append(prob.loss(reverse_greedy(k, prob.p, prob.cost, prob.dist)))
        results["LFU"].append(prob.loss(rng.sample(range(M), k)))
    return results


def offline(runs):
    """Panel (b): suboptimality gap of caches learned from logged datasets of different lengths."""
    prob = problem(M)
    names = ["CLCB-SC", "CUCB-SC", "Epsilon-Greedy", "LFU"]
    gaps = {name: [] for name in names}
    for length in OFFLINE_LENGTHS:
        runs_gaps = {name: [] for name in names}
        for seed in range(runs):
            py_rng, rng = random.Random(seed), np.random.RandomState(seed)
            stream, costs = logged_dataset(prob, length, K, py_rng, rng)
            caches = offline_caches(prob, K, stream, costs, DELTA, 0.2, py_rng, rng)
            for name, cache in caches.items():
                runs_gaps[name].append(prob.loss(cache) - prob.optimal_loss(K))
        for name in names:
            gaps[name].append(runs_gaps[name])
    return {"lengths": OFFLINE_LENGTHS, "gaps": gaps}


def online(runs):
    """Panels (c) and (f): average regret over time, cache switches and runtime per round."""
    results = {"average_regret": {}, "switches": {}, "runtime": {}}
    for name in ONLINE_ALGORITHMS:
        regret, switches, runtime = [], [], []
        for seed in range(runs):
            gaps, n_switches, times = run_online(name, problem(M), K, T, DELTA, seed, explore=0.15)
            regret.append((np.cumsum(gaps) / np.arange(1, T + 1)).tolist())
            switches.append(int(n_switches))
            runtime.append(float(np.mean(times)))
        results["average_regret"][name] = regret
        results["switches"][name] = switches
        results["runtime"][name] = runtime
    return results


def final_gaps(settings, runs):
    """Suboptimality gap of the cache in the last round, for each (m, k) setting and run."""
    results = {name: [] for name in ONLINE_ALGORITHMS}
    for m, k in settings:
        for name in ONLINE_ALGORITHMS:
            results[name].append([run_online(name, problem(m), k, T, DELTA, seed, explore=0.2)[0][-1]
                                  for seed in range(runs)])
    return results


def cache_size(runs):
    """Panel (d): final suboptimality gap for cache sizes k = 5, 7, 10."""
    values = [5, 7, 10]
    return {"values": values, "final_gap": final_gaps([(M, k) for k in values], runs)}


def num_queries(runs):
    """Panel (e): final suboptimality gap for m = 20, 25, 30 queries."""
    values = [20, 25, 30]
    return {"values": values, "final_gap": final_gaps([(m, K) for m in values], runs)}


EXPERIMENTS = {"optimal": optimal, "offline": offline, "online": online,
               "cache_size": cache_size, "num_queries": num_queries}


def summarize(name, results):
    if name == "optimal":
        for key in ("Brute Force", "Reverse Greedy", "LFU"):
            print(f"  {key:15s}", "  ".join(f"k={k}: {v:.4f}" for k, v in zip(results["cache_sizes"], results[key])))
    elif name == "offline":
        for alg, gaps in results["gaps"].items():
            print(f"  {alg:15s} mean gap over lengths {np.mean(gaps):.4f}")
    elif name == "online":
        print(f"  {'':15s} {'final avg regret':>16s} {'switches':>9s} {'runtime (s)':>12s}")
        for alg, regret in results["average_regret"].items():
            print(f"  {alg:15s} {np.mean([r[-1] for r in regret]):16.4f} "
                  f"{np.mean(results['switches'][alg]):9.1f} {np.mean(results['runtime'][alg]):12.5f}")
    else:
        for alg, gaps in results["final_gap"].items():
            print(f"  {alg:15s}", "  ".join(f"{v}: {np.mean(g):.4f}" for v, g in zip(results["values"], gaps)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", metavar="experiment",
                        help=f"any of {', '.join(EXPERIMENTS)} (default: all)")
    parser.add_argument("--runs", type=int, default=10, help="number of random seeds")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.experiments or EXPERIMENTS:
        print(f"Running {name}...")
        results = EXPERIMENTS[name](args.runs)
        (args.out / f"{name}.json").write_text(json.dumps(results))
        summarize(name, results)

    if all((args.out / f"{name}.json").exists() for name in EXPERIMENTS):
        from plot import make_figure
        print(f"Saved {make_figure(args.out)}")


if __name__ == "__main__":
    main()
