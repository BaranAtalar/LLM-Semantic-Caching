# Semantic Caching for Low-Cost LLM Serving

Code for the experiments in [Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation](https://ieeexplore.ieee.org/abstract/document/11571467) (IEEE INFOCOM 2026).

A semantic cache answers an incoming query with the stored response of a similar cached query, which costs much less than calling the LLM but incurs a mismatch cost. Which *k* queries the cache should keep depends on how often each query arrives and how expensive it is to serve, and both are usually unknown. The paper learns the cache offline from logged data (CLCB-SC, CUCB-SC) and online while serving queries (CLCB-SC, CUCB-SC, and the low-switching CLCB-SC-LS).

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the `paraphrase-MiniLM-L6-v2` sentence embedding model and the GPT-2 tokenizer from Hugging Face.

## Reproducing the results

```bash
python run.py
```

This runs all five experiments with the paper's settings and random seeds, prints a summary of each, and saves the results to `results/<experiment>.json` and the six-panel figure to `results/figure.pdf`. It takes a few minutes on a laptop CPU. To run only some experiments, name them, for example `python run.py online cache_size`. `python plot.py` redraws the figure from saved results.

| Experiment | Panel | What it measures |
|---|---|---|
| `optimal` | (a) | Loss of the best cache (brute force), of Reverse Greedy and of LFU, for cache sizes *k* = 5, 7, 10 |
| `offline` | (b) | Suboptimality gap of caches learned from logged datasets of 300 to 600 queries |
| `online` | (c), (f) | Average regret over *T* = 200 rounds, number of cache switches, and time per round |
| `cache_size` | (d) | Suboptimality gap in the last round for *k* = 5, 7, 10 |
| `num_queries` | (e) | Suboptimality gap in the last round for *m* = 20, 25, 30 queries |

Every experiment is repeated over 10 random seeds (`--runs` changes this), and the figure shows means with one standard deviation.

## Setting

* **Queries.** The first *m* queries of `queries.txt` (*m* = 20 unless varied), all equally likely to arrive.
* **Mismatch cost.** The Euclidean distance between the queries' sentence embeddings, min-max normalized to [0, 1].
* **LLM cost.** The query's GPT-2 token count, min-max normalized to [0, 1]. An algorithm observes it, with Gaussian noise (standard deviation 0.05) clipped to [0, 1], only when it sends the query to the LLM.
* **Loss of a cache *M*.** Each query is answered by the cheaper of its closest cached query and the LLM, so loss(*M*) = Σ<sub>*q*</sub> *p*(*q*) · min(*d*(*q*, *M*), *c*(*q*)). The suboptimality gap is the loss minus that of the best cache of the same size, and the regret averages it over rounds.
* **Parameters.** Cache size *k* = 5, confidence parameter δ = 0.8, horizon *T* = 200.

## Algorithms

All methods except LFU choose the cache with **Reverse Greedy**: starting from all queries, repeatedly remove the one whose removal increases the estimated loss the least.

| Algorithm | Cost estimate used by Reverse Greedy |
|---|---|
| CLCB-SC | lower confidence bound on each query's cost |
| CUCB-SC | upper confidence bound on each query's cost |
| CLCB-SC-LS | like CLCB-SC, but the cache is only recomputed when a query's current stage ends, which keeps cache switches low |
| Epsilon-Greedy | empirical mean cost, and each removal is random with probability ε (0.2, or 0.15 in the `online` experiment). Its cost observations have noise 0.1. |
| LFU | keeps the *k* most frequently seen queries |

## Code layout

| File | Contents |
|---|---|
| `caching.py` | problem setup, loss, Reverse Greedy, and the offline and online algorithms |
| `run.py` | the experiments, with their settings at the top of the file |
| `plot.py` | the results figure |
| `queries.txt` | the 50 queries; the experiments use the first 20 to 30 |

## Notes

* In the original experiment code, the offline CLCB-SC and CUCB-SC only let a cached query answer itself (exact matching). They now use the semantic loss above, like every other method, so their curves in panel (b) differ slightly from the paper's figure.
* LFU's choice among equally frequent queries is arbitrary. In panel (a) all queries are equally likely, so its cache is drawn at random; elsewhere ties go to the query listed first in `queries.txt`. Its bars in panels (a) and (e) can therefore differ from the paper's.
* The runtimes in panel (f) depend on the machine. This implementation is faster than the one used for the paper, but the relative speed of the algorithms is the same.

## Citation

```bibtex
@inproceedings{liu2026semantic,
  author    = {Liu, Xutong and Atalar, Baran and Dai, Xiangxiang and Zuo, Jinhang and Wang, Siwei and Lui, John C.S. and Chen, Wei and Joe-Wong, Carlee},
  title     = {Semantic Caching for Low-Cost {LLM} Serving: From Offline Learning to Online Adaptation},
  booktitle = {IEEE INFOCOM 2026 - IEEE Conference on Computer Communications},
  year      = {2026},
  pages     = {1-10},
  doi       = {10.1109/INFOCOM59046.2026.11571467}
}
```
