#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark encoder cache policies with real vLLM serving.

This script sends multimodal requests (with images) to a running vLLM server,
where image references follow a configurable distribution (e.g., Zipf) to
test cache replacement policy effectiveness.

Modes:
    --prepare-images: Generate a synthetic image database
    (default): Send requests and measure TTFT, throughput, etc.
    --compare: Compare results from multiple policy runs

Usage:
    # 1. Prepare images
    python benchmarks/benchmark_encoder_cache_serving.py \
        --prepare-images --image-dir /tmp/images --num-images 20

    # 2. Run benchmark (server must be running)
    python benchmarks/benchmark_encoder_cache_serving.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --port 8000 --image-dir /tmp/images \
        --num-prompts 100 --zipf-alpha 1.2

    # 3. Compare results
    python benchmarks/benchmark_encoder_cache_serving.py \
        --compare --results-dir ./logs/cache_benchmark
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


PROMPTS = [
    "What is shown in this image?",
    "Describe the contents of this image in detail.",
    "What objects can you see in this image?",
    "What is happening in this image?",
    "Can you describe the colors and shapes in this image?",
]


def prepare_images(image_dir: str, num_images: int, seed: int = 42):
    """Generate synthetic images for benchmarking."""
    if Image is None:
        print("Pillow is required for image generation. "
              "Install with: pip install Pillow")
        sys.exit(1)

    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Generate images of varying sizes to simulate realistic workloads
    sizes = [(224, 224), (336, 336), (448, 448), (512, 512), (640, 480)]

    for i in range(num_images):
        w, h = sizes[i % len(sizes)]
        # Create a colored image with some patterns
        pixels = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, "RGB")

        path = os.path.join(image_dir, f"img_{i:04d}.jpg")
        img.save(path, "JPEG", quality=85)

    print(f"Created {num_images} images in {image_dir}")
    for i, (w, h) in enumerate(sizes):
        count = sum(1 for j in range(num_images) if j % len(sizes) == i)
        print(f"  {w}x{h}: {count} images")


def generate_request_sequence(
    num_images: int,
    num_prompts: int,
    distribution: str = "zipf",
    zipf_alpha: float = 1.2,
    seed: int = 42,
) -> list[int]:
    """Generate a sequence of image indices following the given distribution."""
    rng = np.random.RandomState(seed)

    if distribution == "zipf":
        ranks = np.arange(1, num_images + 1, dtype=float)
        probs = 1.0 / (ranks ** zipf_alpha)
    elif distribution == "uniform":
        probs = np.ones(num_images)
    elif distribution == "bimodal":
        probs = np.ones(num_images)
        probs[: num_images // 2] = 10.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    probs /= probs.sum()
    return rng.choice(num_images, size=num_prompts, p=probs).tolist()


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    image_url: str,
    prompt: str,
    request_id: int,
) -> dict:
    """Send a single chat completion request with an image."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 50,
        "stream": False,
    }

    start_time = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            end_time = time.perf_counter()
            result = await resp.json()

            ttft = end_time - start_time
            success = resp.status == 200

            output_tokens = 0
            if success and "usage" in result:
                output_tokens = result["usage"].get("completion_tokens", 0)

            return {
                "request_id": request_id,
                "success": success,
                "ttft": ttft,
                "output_tokens": output_tokens,
                "status": resp.status,
            }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "ttft": end_time - start_time,
            "output_tokens": 0,
            "error": str(e),
        }


async def run_benchmark(
    model: str,
    port: int,
    image_dir: str,
    image_indices: list[int],
    concurrency: int = 4,
) -> list[dict]:
    """Run the benchmark by sending requests sequentially or with concurrency."""
    url = f"http://localhost:{port}/v1/chat/completions"
    rng = np.random.RandomState(0)

    # Load image paths
    image_files = sorted(glob.glob(os.path.join(image_dir, "img_*.jpg")))
    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    # Pre-encode images as base64
    print(f"Encoding {len(image_files)} images as base64...")
    image_b64 = {i: image_to_base64(f) for i, f in enumerate(image_files)}

    results = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(session, idx, image_idx):
        async with sem:
            prompt = PROMPTS[idx % len(PROMPTS)]
            return await send_request(
                session, url, model, image_b64[image_idx], prompt, idx
            )

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            bounded_request(session, i, img_idx)
            for i, img_idx in enumerate(image_indices)
        ]

        total = len(tasks)
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total}")

    return results


def analyze_results(results: list[dict], policy_name: str) -> dict:
    """Analyze benchmark results and compute summary statistics."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "policy": policy_name,
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "error": "No successful requests",
        }

    ttfts = [r["ttft"] for r in successful]
    output_tokens = [r["output_tokens"] for r in successful]

    return {
        "policy": policy_name,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "ttft_mean": float(np.mean(ttfts)),
        "ttft_median": float(np.median(ttfts)),
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p90": float(np.percentile(ttfts, 90)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
        "ttft_min": float(np.min(ttfts)),
        "ttft_max": float(np.max(ttfts)),
        "total_time": float(np.sum(ttfts)),
        "avg_output_tokens": float(np.mean(output_tokens)),
        "throughput_rps": len(successful) / float(np.sum(ttfts))
        if np.sum(ttfts) > 0
        else 0.0,
    }


def compare_results(results_dir: str, timestamp: str | None = None):
    """Compare results from different policy runs."""
    pattern = os.path.join(results_dir, "results_*")
    if timestamp:
        pattern = os.path.join(results_dir, f"results_*_{timestamp}.json")

    result_files = sorted(glob.glob(pattern))
    if not result_files:
        print(f"No result files found matching {pattern}")
        return

    summaries = []
    for f in result_files:
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "summary" in data:
            summaries.append(data["summary"])
        elif isinstance(data, dict):
            summaries.append(data)

    if len(summaries) < 2:
        print("Need at least 2 result files to compare")
        for s in summaries:
            print(f"\n  Policy: {s.get('policy', 'unknown')}")
            for k, v in s.items():
                if k != "policy":
                    print(f"    {k}: {v}")
        return

    # Print comparison table
    print(f"\n{'Metric':<25}", end="")
    for s in summaries:
        print(f"  {s.get('policy', '?'):<15}", end="")
    print()
    print("-" * (25 + 17 * len(summaries)))

    metrics = [
        "successful", "failed", "ttft_mean", "ttft_median",
        "ttft_p90", "ttft_p99", "throughput_rps",
    ]
    for metric in metrics:
        print(f"  {metric:<23}", end="")
        for s in summaries:
            val = s.get(metric, "N/A")
            if isinstance(val, float):
                print(f"  {val:<15.4f}", end="")
            else:
                print(f"  {val!s:<15}", end="")
        print()

    # Improvement calculation
    if len(summaries) == 2:
        lru = next((s for s in summaries if s.get("policy") == "lru"), None)
        od = next(
            (s for s in summaries if s.get("policy") == "online_dual"), None
        )
        if lru and od and lru.get("ttft_mean") and od.get("ttft_mean"):
            ttft_improvement = (
                (lru["ttft_mean"] - od["ttft_mean"])
                / lru["ttft_mean"]
                * 100
            )
            print(f"\n  TTFT improvement (OnlineDual vs LRU): "
                  f"{ttft_improvement:+.2f}%")
            if lru.get("throughput_rps") and od.get("throughput_rps"):
                tput_improvement = (
                    (od["throughput_rps"] - lru["throughput_rps"])
                    / lru["throughput_rps"]
                    * 100
                )
                print(f"  Throughput improvement: {tput_improvement:+.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark encoder cache policies with vLLM serving"
    )

    # Mode selection
    parser.add_argument(
        "--prepare-images", action="store_true",
        help="Generate synthetic image database"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare results from multiple runs"
    )

    # Server settings
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)

    # Image settings
    parser.add_argument("--image-dir", type=str, default="/tmp/vllm_bench_images")
    parser.add_argument("--num-images", type=int, default=20)

    # Benchmark settings
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--distribution", type=str, default="zipf",
                        choices=["zipf", "uniform", "bimodal"])
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--policy-name", type=str, default="unknown")
    parser.add_argument("--results-dir", type=str, default="./logs/cache_benchmark")
    parser.add_argument("--timestamp", type=str, default=None)

    args = parser.parse_args()

    if args.prepare_images:
        prepare_images(args.image_dir, args.num_images, args.seed)
        return

    if args.compare:
        compare_results(args.results_dir, args.timestamp)
        return

    # Generate request sequence
    image_indices = generate_request_sequence(
        num_images=args.num_images,
        num_prompts=args.num_prompts,
        distribution=args.distribution,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )

    # Print distribution info
    from collections import Counter
    counts = Counter(image_indices)
    print(f"\nRequest distribution ({args.distribution}, "
          f"alpha={args.zipf_alpha}):")
    print(f"  Total requests: {args.num_prompts}")
    print(f"  Unique images referenced: {len(counts)}/{args.num_images}")
    top5 = counts.most_common(5)
    print(f"  Top 5 images: {top5}")
    print()

    # Run benchmark
    print(f"Running benchmark with policy={args.policy_name}...")
    results = asyncio.run(
        run_benchmark(
            model=args.model,
            port=args.port,
            image_dir=args.image_dir,
            image_indices=image_indices,
            concurrency=args.concurrency,
        )
    )

    # Analyze
    summary = analyze_results(results, args.policy_name)
    print(f"\n{'='*50}")
    print(f"Results for policy: {args.policy_name}")
    print(f"{'='*50}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    if args.output_json:
        output = {
            "summary": summary,
            "config": {
                "model": args.model,
                "num_prompts": args.num_prompts,
                "num_images": args.num_images,
                "distribution": args.distribution,
                "zipf_alpha": args.zipf_alpha,
                "seed": args.seed,
                "concurrency": args.concurrency,
            },
            "raw_results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
