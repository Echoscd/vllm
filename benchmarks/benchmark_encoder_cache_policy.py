#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark encoder cache replacement policies: LRU vs OnlineDual.

This script simulates multimodal request traffic with configurable
request distributions and compares cache hit rates, total compute cost,
and other metrics between the LRU and OnlineDual cache policies.

Usage:
    python benchmarks/benchmark_encoder_cache_policy.py \
        --num-items 50 --cache-size 200 --num-rounds 5000 \
        --distribution zipf --zipf-alpha 1.2

The script generates synthetic workloads where each "request" references
one or more multimodal items sampled from a fixed distribution, and
measures how well each cache policy avoids recomputation.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Add parent directory so we can import from vllm
sys.path.insert(0, ".")

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    OnlineDualEncoderCacheManager,
)


@dataclass
class SimItem:
    """A multimodal item type in the simulation."""
    item_id: str
    num_tokens: int  # memory cost m_i
    compute_cost: float  # compute cost c_i (seconds)
    probability: float  # p_i: probability of being requested


class MockRequest:
    """Mock request for simulation."""

    def __init__(self, request_id: str, mm_hashes: list[str],
                 token_counts: list[int]):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(
                    offset=0, length=self._token_counts[i]
                ),
            )
            self.mm_features.append(feature)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self._token_counts[input_id]


def generate_items(
    num_items: int,
    distribution: str,
    zipf_alpha: float = 1.2,
    min_tokens: int = 50,
    max_tokens: int = 500,
    min_cost: float = 0.01,
    max_cost: float = 0.2,
    seed: int = 42,
) -> list[SimItem]:
    """Generate K multimodal item types with given distribution."""
    rng = np.random.RandomState(seed)

    # Generate item properties
    tokens = rng.randint(min_tokens, max_tokens + 1, size=num_items)
    costs = rng.uniform(min_cost, max_cost, size=num_items)

    # Generate probabilities
    if distribution == "zipf":
        ranks = np.arange(1, num_items + 1, dtype=float)
        probs = 1.0 / (ranks ** zipf_alpha)
    elif distribution == "uniform":
        probs = np.ones(num_items)
    elif distribution == "bimodal":
        # Half items are popular, half are rare
        probs = np.ones(num_items)
        probs[: num_items // 2] = 10.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    probs /= probs.sum()

    return [
        SimItem(
            item_id=f"item_{i}",
            num_tokens=int(tokens[i]),
            compute_cost=float(costs[i]),
            probability=float(probs[i]),
        )
        for i in range(num_items)
    ]


def run_simulation(
    items: list[SimItem],
    cache_size: int,
    num_rounds: int,
    policy: str,
    seed: int = 0,
    soft_budget_ratio: float = 0.9,
    eta: float = 0.001,
) -> dict:
    """Run a cache simulation with the given policy.

    Returns metrics dict with hit rate, total cost, etc.
    """
    rng = np.random.RandomState(seed)
    probs = np.array([item.probability for item in items])

    # Create cache manager
    if policy == "lru":
        cache = EncoderCacheManager(cache_size=cache_size)
    elif policy == "online_dual":
        cache = OnlineDualEncoderCacheManager(
            cache_size=cache_size,
            soft_budget_ratio=soft_budget_ratio,
            initial_eta=eta,
            default_compute_cost=np.mean([it.compute_cost for it in items]),
        )
        # Pre-record compute costs
        for item in items:
            cache.record_compute_cost(item.item_id, item.compute_cost)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Metrics
    hits = 0
    misses = 0
    total_compute_cost = 0.0
    evictions = 0

    for t in range(num_rounds):
        # Sample a request
        item_idx = rng.choice(len(items), p=probs)
        item = items[item_idx]
        request_id = f"req_{t}"
        req = MockRequest(request_id, [item.item_id], [item.num_tokens])

        # Check cache
        if cache.check_and_update_cache(req, 0):
            hits += 1
            # Free reference immediately (simulate short-lived request)
            cache.free_encoder_input(req, 0)
        else:
            misses += 1
            total_compute_cost += item.compute_cost

            # Try to allocate
            if cache.can_allocate(req, 0, int(1e9), 0):
                cache.allocate(req, 0)
                # Free reference
                cache.free_encoder_input(req, 0)

            freed = cache.get_freed_mm_hashes()
            evictions += len(freed)

    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

    result = {
        "policy": policy,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "total_compute_cost": total_compute_cost,
        "evictions": evictions,
        "cache_size": cache_size,
        "num_rounds": num_rounds,
        "num_items": len(items),
    }

    if policy == "online_dual" and isinstance(cache, OnlineDualEncoderCacheManager):
        result.update(cache.get_stats())

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark encoder cache replacement policies"
    )
    parser.add_argument(
        "--num-items", type=int, default=50,
        help="Number of unique multimodal item types (K)"
    )
    parser.add_argument(
        "--cache-size", type=int, default=200,
        help="Cache capacity in encoder tokens (B)"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5000,
        help="Number of request rounds (T)"
    )
    parser.add_argument(
        "--distribution", type=str, default="zipf",
        choices=["zipf", "uniform", "bimodal"],
        help="Request probability distribution"
    )
    parser.add_argument(
        "--zipf-alpha", type=float, default=1.2,
        help="Zipf distribution parameter (higher = more skewed)"
    )
    parser.add_argument(
        "--min-tokens", type=int, default=50,
        help="Minimum tokens per multimodal item"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Maximum tokens per multimodal item"
    )
    parser.add_argument(
        "--soft-budget-ratio", type=float, default=0.9,
        help="Soft budget as fraction of cache size for OnlineDual"
    )
    parser.add_argument(
        "--eta", type=float, default=0.001,
        help="Learning rate for OnlineDual lambda update"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for item generation"
    )
    parser.add_argument(
        "--sim-seed", type=int, default=0,
        help="Random seed for simulation"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write JSON results"
    )
    parser.add_argument(
        "--sweep-cache-sizes", type=str, default=None,
        help="Comma-separated cache sizes to sweep (e.g., '100,200,500,1000')"
    )

    args = parser.parse_args()

    # Generate items
    items = generate_items(
        num_items=args.num_items,
        distribution=args.distribution,
        zipf_alpha=args.zipf_alpha,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    total_item_tokens = sum(it.num_tokens for it in items)
    print(f"Generated {len(items)} item types, "
          f"total tokens: {total_item_tokens}")
    print(f"Distribution: {args.distribution}"
          + (f" (alpha={args.zipf_alpha})" if args.distribution == "zipf"
             else ""))
    print()

    all_results = []

    if args.sweep_cache_sizes:
        cache_sizes = [int(x) for x in args.sweep_cache_sizes.split(",")]
    else:
        cache_sizes = [args.cache_size]

    for cache_size in cache_sizes:
        print(f"{'='*60}")
        print(f"Cache size: {cache_size} tokens "
              f"({cache_size/total_item_tokens*100:.1f}% of total)")
        print(f"{'='*60}")

        for policy in ["lru", "online_dual"]:
            start = time.perf_counter()
            result = run_simulation(
                items=items,
                cache_size=cache_size,
                num_rounds=args.num_rounds,
                policy=policy,
                seed=args.sim_seed,
                soft_budget_ratio=args.soft_budget_ratio,
                eta=args.eta,
            )
            elapsed = time.perf_counter() - start

            result["sim_time_seconds"] = elapsed
            all_results.append(result)

            print(f"\n  Policy: {policy}")
            print(f"  Hit rate:        {result['hit_rate']:.4f} "
                  f"({result['hits']}/{result['hits']+result['misses']})")
            print(f"  Compute cost:    {result['total_compute_cost']:.4f}s")
            print(f"  Evictions:       {result['evictions']}")
            print(f"  Sim time:        {elapsed:.3f}s")

        # Compare
        lru_result = [r for r in all_results
                      if r["policy"] == "lru"
                      and r["cache_size"] == cache_size][0]
        od_result = [r for r in all_results
                     if r["policy"] == "online_dual"
                     and r["cache_size"] == cache_size][0]

        hit_improvement = (
            (od_result["hit_rate"] - lru_result["hit_rate"])
            / max(lru_result["hit_rate"], 1e-9) * 100
        )
        cost_reduction = (
            (lru_result["total_compute_cost"] - od_result["total_compute_cost"])
            / max(lru_result["total_compute_cost"], 1e-9) * 100
        )

        print(f"\n  --- Comparison (cache_size={cache_size}) ---")
        print(f"  Hit rate improvement:  {hit_improvement:+.2f}%")
        print(f"  Compute cost reduction: {cost_reduction:+.2f}%")
        print()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
