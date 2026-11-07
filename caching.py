"""Semantic cache model, Reverse Greedy, and the offline and online caching algorithms.

A cache M holds k of the m queries. A query q that arrives is either answered by the closest
cached query, at mismatch cost d(q, M), or sent to the LLM at cost c(q), whichever is cheaper.
The expected loss of a cache is

    loss(M) = sum_q p(q) * min(d(q, M), c(q)),

where p is the query distribution, d the min-max normalized embedding distance and c the
min-max normalized token count of the query.
"""
import math
import random
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

QUERIES = Path(__file__).resolve().parent.joinpath("queries.txt").read_text().splitlines()
SIGMA = 0.05  # standard deviation of the observed cost noise; also scales the confidence radius


@lru_cache(maxsize=None)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")


@lru_cache(maxsize=None)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def embed(queries):
    return _encoder().encode(queries)


def count_tokens(queries):
    return [len(_tokenizer().encode(q)) for q in queries]


def min_max(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min())


def cache_loss(cache, p, cost, dist):
    """sum_q p(q) * min(distance from q to its closest cached query, cost(q))."""
    nearest = dist[:, sorted(cache)].min(axis=1)
    return float(np.sum(p * np.minimum(nearest, cost)))


def reverse_greedy(k, p, cost, dist, explore=0.0, rng=None, py_rng=None):
    """Start from all queries and repeatedly drop the one whose removal increases the loss least.

    With `explore` > 0 (Epsilon-Greedy), each drop is instead a uniformly random query with
    probability `explore`.
    """
    cache = set(range(len(p)))
    while len(cache) > k:
        if explore > 0 and rng.rand() < explore:
            cache.remove(py_rng.choice(list(cache)))
        else:
            candidates = list(cache)
            losses = [cache_loss(cache - {q}, p, cost, dist) for q in candidates]
            # Rounding makes exact ties (common, e.g. for queries never seen) break the same way
            # on every machine instead of depending on floating-point summation order.
            cache.remove(candidates[int(np.argmin(np.round(losses, 12)))])
    return cache


class Problem:
    """The first m queries of queries.txt, arriving uniformly at random."""

    def __init__(self, m):
        queries = QUERIES[:m]
        self.m = m
        self.tokens = np.array(count_tokens(queries))
        self.cost = min_max(self.tokens)
        self.dist = min_max(euclidean_distances(embed(queries)))
        self.p = np.full(m, 1 / m)
        self._optimum = {}

    def loss(self, cache):
        return cache_loss(cache, self.p, self.cost, self.dist)

    def optimal_loss(self, k):
        """Loss of the best cache of size k, found by brute force."""
        if k not in self._optimum:
            self._optimum[k] = min(self.loss(c) for c in combinations(range(self.m), k))
        return self._optimum[k]

    def nearest_distance(self, q, cache):
        return self.dist[q, sorted(cache)].min()


def confidence_radius(observations, m, horizon, delta):
    return SIGMA * np.sqrt(2 * math.log(4 * m * horizon / delta) / observations)


def lfu(k, counts):
    """The k most frequent queries (ties broken by query index)."""
    return set(np.argsort(-np.asarray(counts), kind="stable")[:k].tolist())


# ---------------------------------------------------------------------------------------------
# Offline: learn a cache from a logged dataset
# ---------------------------------------------------------------------------------------------

def logged_dataset(problem, length, k, py_rng, rng):
    """Queries and costs logged under a behaviour cache that starts random and, on a miss,
    admits the query with probability 1/2 by evicting a random entry. The LLM cost of a query
    (with noise) is only logged when the query was not in the cache."""
    m = problem.m
    stream = [py_rng.randrange(m) for _ in range(length)]
    py_rng.sample(range(m), k)  # unused draw, kept so seeds reproduce the paper's experiments
    cache = set(py_rng.sample(range(m), k))
    logged_caches = []
    for q in stream:
        if q not in cache and py_rng.random() < 0.5:
            cache.discard(py_rng.choice(list(cache)))
            cache.add(q)
        logged_caches.append(set(cache))

    true_costs = min_max(problem.tokens[stream])
    costs = [0.0 if q in cached else np.clip(rng.normal(c, SIGMA), 0, 1)
             for q, c, cached in zip(stream, true_costs, logged_caches)]
    return np.array(stream), np.array(costs)


def offline_estimates(stream, costs, m, delta):
    """Empirical arrival probabilities and mean costs with lower and upper confidence bounds."""
    n = len(stream)
    counts = np.bincount(stream, minlength=m)
    observed = np.bincount(stream, weights=(costs > 0).astype(float), minlength=m)
    total = np.bincount(stream, weights=costs, minlength=m)
    seen = observed > 0
    c_hat = np.divide(total, observed, out=np.zeros(m), where=seen)
    radius = confidence_radius(np.maximum(observed, 1), m, n, delta)
    lcb = np.where(seen, np.maximum(c_hat - radius, 0), 0.0)
    ucb = np.where(seen, c_hat + radius, 1.0)
    # The paper's experiments used these estimates rounded to 4 decimals.
    p_hat, c_hat, lcb, ucb = (np.round(x, 4) for x in (counts / n, c_hat, lcb, ucb))
    return p_hat, c_hat, lcb, ucb, counts


def offline_caches(problem, k, stream, costs, delta, explore, py_rng, rng):
    """Caches chosen by each offline method from one logged dataset."""
    p_hat, c_hat, lcb, ucb, counts = offline_estimates(stream, costs, problem.m, delta)
    return {
        "CLCB-SC": reverse_greedy(k, p_hat, lcb, problem.dist),
        "CUCB-SC": reverse_greedy(k, p_hat, ucb, problem.dist),
        "Epsilon-Greedy": reverse_greedy(k, p_hat, c_hat, problem.dist, explore, rng, py_rng),
        "LFU": lfu(k, counts),
    }


# ---------------------------------------------------------------------------------------------
# Online: learn the cache while serving a stream of queries
# ---------------------------------------------------------------------------------------------

class OnlineCache:
    """Keeps arrival counts and observed LLM costs. Each round `select` returns the cache, and
    `cost_estimate[q]` decides whether the arriving query q is sent to the LLM (observing its cost)
    or answered from the cache."""

    noise = SIGMA

    def __init__(self, problem, k, horizon, delta):
        self.problem, self.k, self.horizon, self.delta = problem, k, horizon, delta
        m = problem.m
        self.arrivals = np.zeros(m)
        self.observations = np.zeros(m)
        self.total_cost = np.zeros(m)
        self.cost_estimate = np.zeros(m)

    def observe(self, q, cost):
        self.observations[q] += 1
        self.total_cost[q] += cost

    def estimates(self, t):
        p_hat = self.arrivals / (t + 1)
        seen = self.observations > 0
        c_hat = np.divide(self.total_cost, self.observations, out=np.zeros(len(seen)), where=seen)
        return p_hat, c_hat, seen

    def radius(self, horizon):
        if not self.observations.any():  # nothing observed yet (always the case in round 0)
            return np.zeros(self.problem.m)
        return confidence_radius(np.maximum(self.observations, 1), self.problem.m, horizon, self.delta)

    def lower_confidence_costs(self, t, horizon):
        p_hat, c_hat, seen = self.estimates(t)
        return p_hat, np.where(seen, np.maximum(c_hat - self.radius(horizon), 0), 0.0)

    def reverse_greedy(self, p_hat):
        return reverse_greedy(self.k, p_hat, self.cost_estimate, self.problem.dist)


class CLCB(OnlineCache):
    """CLCB-SC: every round, Reverse Greedy with lower confidence bounds on the costs."""

    def select(self, t, q):
        p_hat, self.cost_estimate = self.lower_confidence_costs(t, self.horizon)
        return self.reverse_greedy(p_hat)


class CUCB(OnlineCache):
    """CUCB-SC: every round, Reverse Greedy with upper confidence bounds on the costs."""

    def select(self, t, q):
        p_hat, c_hat, seen = self.estimates(t)
        self.cost_estimate = np.where(seen, np.minimum(c_hat + self.radius(t), 1), 1.0)
        return self.reverse_greedy(p_hat)


class CLCBLowSwitch(OnlineCache):
    """CLCB-SC-LS: CLCB-SC that only recomputes the cache when a query starts a new stage.

    A query's stage ends once its arrivals in the stage reach 1 + sqrt(k * T * N / m), where N
    is its number of arrivals in earlier stages.
    """

    def __init__(self, problem, k, horizon, delta):
        super().__init__(problem, k, horizon, delta)
        self.stage_arrivals = np.zeros(problem.m)
        self.past_arrivals = np.zeros(problem.m)
        self.cache = None

    def select(self, t, q):
        self.stage_arrivals[q] += 1
        stage_length = 1 + np.sqrt(self.k * self.horizon * self.past_arrivals / self.problem.m)
        finished = self.stage_arrivals >= stage_length
        if finished.any():
            self.past_arrivals[finished] += self.stage_arrivals[finished]
            self.stage_arrivals[finished] = 0
            p_hat, self.cost_estimate = self.lower_confidence_costs(t, t)
            self.cache = self.reverse_greedy(p_hat)
        return self.cache


class EpsilonGreedy(OnlineCache):
    """Reverse Greedy with the empirical mean costs; each removal is random with probability eps."""

    noise = 0.1

    def __init__(self, problem, k, horizon, delta, explore, rng, py_rng):
        super().__init__(problem, k, horizon, delta)
        self.explore, self.rng, self.py_rng = explore, rng, py_rng
        rng.standard_normal(horizon)  # unused draws, kept so seeds reproduce the paper's experiments

    def select(self, t, q):
        p_hat, self.cost_estimate, _ = self.estimates(t)
        return reverse_greedy(self.k, p_hat, self.cost_estimate, self.problem.dist,
                              self.explore, self.rng, self.py_rng)


ONLINE_ALGORITHMS = {"CLCB-SC": CLCB, "CUCB-SC": CUCB, "CLCB-SC-LS": CLCBLowSwitch,
                     "Epsilon-Greedy": EpsilonGreedy}


def run_online(name, problem, k, horizon, delta, seed, explore):
    """One run of an online algorithm. Returns the suboptimality gap of every round, the number
    of cache switches and the time spent choosing the cache in each round."""
    py_rng, rng = random.Random(seed), np.random.RandomState(seed)
    stream = [py_rng.randrange(problem.m) for _ in range(horizon)]
    true_costs = min_max(problem.tokens[stream])
    optimum = problem.optimal_loss(k)

    if name == "LFU":
        gap = problem.loss(lfu(k, np.bincount(stream, minlength=problem.m))) - optimum
        return [gap] * horizon, 0, [0.0] * horizon

    extra = dict(explore=explore, rng=rng, py_rng=py_rng) if name == "Epsilon-Greedy" else {}
    algo = ONLINE_ALGORITHMS[name](problem, k, horizon, delta, **extra)
    gaps, runtimes, switches, previous = [], [], 0, None
    for t, q in enumerate(stream):
        start = time.perf_counter()
        algo.arrivals[q] += 1
        cache = algo.select(t, q)
        runtimes.append(time.perf_counter() - start)

        gaps.append(problem.loss(cache) - optimum)
        switches += previous is not None and cache != previous
        previous = set(cache)
        if problem.nearest_distance(q, cache) > algo.cost_estimate[q]:
            algo.observe(q, np.clip(rng.normal(true_costs[t], algo.noise), 0, 1))
    return gaps, switches, runtimes
