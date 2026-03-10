#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Encoder 架构下的缓存策略压力测试工具。

功能:
  --prepare-images : 生成合成图片数据库（不同尺寸/内容）
  (默认)           : 按 Zipf 分布引用图片发请求，采集 TTFT 等指标
  --compare        : 对比多次运行结果

图片通过 file:// 协议传给 vLLM（利用 --allowed-local-media-path），
这样 encoder 端可以直接读取本地文件，与生产环境一致。

用法:
    # 1. 生成图片
    python benchmark_disagg_cache_workload.py \\
        --prepare-images --image-dir /tmp/images --num-images 30

    # 2. 运行测试 (确保 disagg 集群 + proxy 已启动)
    python benchmark_disagg_cache_workload.py \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --proxy-port 10001 --image-dir /tmp/images \\
        --num-prompts 200 --zipf-alpha 1.5

    # 3. 对比
    python benchmark_disagg_cache_workload.py \\
        --compare --results-dir ./logs/disagg_cache_bench
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import aiohttp
except ImportError:
    print("请安装 aiohttp: pip install aiohttp")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None

# ---------- 多样化的提示词，避免 KV-cache 全部命中 ----------
PROMPTS = [
    "What is shown in this image? Describe it briefly.",
    "Describe the contents of this image in one paragraph.",
    "What objects and colors can you see in this image?",
    "What is the most prominent feature in this image?",
    "If you had to give this image a title, what would it be?",
    "Describe the spatial layout of elements in this image.",
    "What mood or atmosphere does this image convey?",
    "List the main visual elements in this image.",
]

# 默认最大重试次数
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # 秒


# ========================== 图片生成 ==========================
def prepare_images(image_dir: str, num_images: int, seed: int = 42):
    """生成不同尺寸和内容的合成图片。"""
    if Image is None:
        print("需要 Pillow: pip install Pillow")
        sys.exit(1)

    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 5 种典型尺寸，模拟真实场景中图片大小差异
    sizes = [
        (224, 224),   # 小图
        (336, 336),   # 中等
        (448, 448),   # 标准
        (512, 512),   # 大图
        (640, 480),   # 宽幅
    ]

    for i in range(num_images):
        w, h = sizes[i % len(sizes)]
        # 不同内容：随机像素 + 不同颜色倾向，确保 mm_hash 不同
        base_color = rng.randint(0, 200, size=3)
        noise = rng.randint(0, 56, (h, w, 3), dtype=np.uint8)
        pixels = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels, "RGB")

        path = os.path.join(image_dir, f"img_{i:04d}.jpg")
        img.save(path, "JPEG", quality=85)

    print(f"已创建 {num_images} 张图片 -> {image_dir}")
    size_counts = Counter(sizes[i % len(sizes)] for i in range(num_images))
    for (w, h), count in sorted(size_counts.items()):
        print(f"  {w}x{h}: {count} 张")


# ======================== 分布生成 ========================
def generate_request_sequence(
    num_images: int,
    num_prompts: int,
    distribution: str = "zipf",
    zipf_alpha: float = 1.5,
    seed: int = 42,
) -> list[int]:
    """按指定分布生成图片引用序列。"""
    rng = np.random.RandomState(seed)

    if distribution == "zipf":
        ranks = np.arange(1, num_images + 1, dtype=float)
        probs = 1.0 / (ranks ** zipf_alpha)
    elif distribution == "uniform":
        probs = np.ones(num_images)
    elif distribution == "bimodal":
        probs = np.ones(num_images)
        probs[: num_images // 4] = 20.0  # 前 25% 的图片非常热门
    else:
        raise ValueError(f"未知分布: {distribution}")

    probs /= probs.sum()
    return rng.choice(num_images, size=num_prompts, p=probs).tolist()


# ======================== 请求发送 ========================
async def send_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    image_path: str,
    prompt: str,
    request_idx: int,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """发送一个带 file:// 图片的 chat completion 请求，带重试。"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{image_path}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 32,
        "stream": False,
    }

    last_error = None
    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                elapsed = time.perf_counter() - start
                body = await resp.json()

                if resp.status == 200:
                    output_tokens = 0
                    if "usage" in body:
                        output_tokens = body["usage"].get(
                            "completion_tokens", 0
                        )
                    return {
                        "idx": request_idx,
                        "success": True,
                        "ttft": elapsed,
                        "status": resp.status,
                        "output_tokens": output_tokens,
                        "attempts": attempt + 1,
                    }

                # 服务端错误 (5xx)，可重试
                last_error = (
                    f"HTTP {resp.status}: "
                    f"{body.get('detail', body.get('error', ''))}"
                )
                if resp.status >= 500 and attempt < max_retries:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue

                # 客户端错误 (4xx) 或重试耗尽
                return {
                    "idx": request_idx,
                    "success": False,
                    "ttft": elapsed,
                    "status": resp.status,
                    "error": last_error,
                    "attempts": attempt + 1,
                }

        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
        ) as e:
            elapsed = time.perf_counter() - start
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
                continue

            return {
                "idx": request_idx,
                "success": False,
                "ttft": elapsed,
                "error": last_error,
                "attempts": attempt + 1,
            }

    # 不应到这里
    return {
        "idx": request_idx,
        "success": False,
        "ttft": 0,
        "error": last_error or "unknown",
        "attempts": max_retries + 1,
    }


async def run_workload(
    model: str,
    proxy_port: int,
    image_dir: str,
    image_indices: list[int],
    concurrency: int = 4,
    request_rate: float = 0.0,
) -> list[dict]:
    """按序列发请求，用 Semaphore 控制并发，可选请求间隔。

    Args:
        request_rate: 每秒发送的请求数。0 表示尽快发送（仅受并发限制）。
    """
    url = f"http://localhost:{proxy_port}/v1/chat/completions"

    image_files = sorted(glob.glob(os.path.join(image_dir, "img_*.jpg")))
    if not image_files:
        print(f"在 {image_dir} 中找不到图片")
        sys.exit(1)

    print(f"  找到 {len(image_files)} 张图片")
    print(f"  并发: {concurrency}, 请求速率: "
          f"{'unlimited' if request_rate <= 0 else f'{request_rate:.1f} req/s'}")

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    interval = 1.0 / request_rate if request_rate > 0 else 0.0
    total = len(image_indices)
    done_count = 0
    lock = asyncio.Lock()
    start_wall = time.perf_counter()

    async def do_request(session, idx, img_idx):
        nonlocal done_count
        async with sem:
            prompt = PROMPTS[idx % len(PROMPTS)]
            img_path = image_files[img_idx]
            result = await send_one_request(
                session, url, model, img_path, prompt, idx
            )
            async with lock:
                done_count += 1
                if done_count % max(1, total // 20) == 0 or done_count == total:
                    elapsed_wall = time.perf_counter() - start_wall
                    print(
                        f"  进度: {done_count}/{total} "
                        f"({elapsed_wall:.1f}s elapsed)"
                    )
            return result

    connector = aiohttp.TCPConnector(limit=concurrency + 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 先发一个 warmup 请求确认服务可用
        print("  发送 warmup 请求...")
        warmup = await send_one_request(
            session, url, model, image_files[0], PROMPTS[0], -1
        )
        if not warmup.get("success"):
            print(f"  [WARN] warmup 失败: {warmup.get('error', warmup)}")
            print("  服务可能未就绪，继续尝试...")
        else:
            print(f"  warmup 成功 (TTFT={warmup['ttft']:.2f}s)")

        # 按顺序提交任务，可选间隔
        tasks = []
        for i, img_idx in enumerate(image_indices):
            task = asyncio.create_task(do_request(session, i, img_idx))
            tasks.append(task)
            if interval > 0:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    return list(results)


# ======================== 结果分析 ========================
def analyze(results: list[dict], policy: str, config: dict) -> dict:
    """计算汇总统计指标。"""
    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]

    if not ok:
        return {"policy": policy, "successful": 0, "failed": len(fail),
                "error": "无成功请求"}

    # 统计错误类型
    if fail:
        error_types: dict[str, int] = {}
        for r in fail:
            err = r.get("error", r.get("status", "unknown"))
            key = str(err)[:80]
            error_types[key] = error_types.get(key, 0) + 1
        print(f"\n  失败请求错误分布 ({len(fail)} 个):")
        for err_key, cnt in sorted(
            error_types.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    [{cnt:>4}] {err_key}")

    ttfts = sorted(r["ttft"] for r in ok)
    out_toks = [r.get("output_tokens", 0) for r in ok]

    # 吞吐量: 用整体墙钟时间（从第一个到最后一个完成）
    wall_times = [r["ttft"] for r in results]
    total_wall = max(wall_times) if wall_times else 0

    summary = {
        "policy": policy,
        "total_requests": len(results),
        "successful": len(ok),
        "failed": len(fail),
        "ttft_mean":   float(np.mean(ttfts)),
        "ttft_median": float(np.median(ttfts)),
        "ttft_p50":    float(np.percentile(ttfts, 50)),
        "ttft_p90":    float(np.percentile(ttfts, 90)),
        "ttft_p95":    float(np.percentile(ttfts, 95)),
        "ttft_p99":    float(np.percentile(ttfts, 99)),
        "ttft_min":    float(np.min(ttfts)),
        "ttft_max":    float(np.max(ttfts)),
        "avg_output_tokens": float(np.mean(out_toks)),
        "throughput_rps": len(ok) / total_wall if total_wall > 0 else 0,
    }
    return summary


def compare_results(results_dir: str, timestamp: str | None = None):
    """对比两次运行的 JSON 结果文件。"""
    pattern = os.path.join(results_dir, "results_*")
    if timestamp:
        pattern = os.path.join(results_dir, f"results_*_{timestamp}.json")

    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        print(f"至少需要 2 个结果文件，找到 {len(files)} 个: {pattern}")
        if files:
            for f in files:
                with open(f) as fp:
                    data = json.load(fp)
                s = data.get("summary", data)
                print(f"\n  策略: {s.get('policy', '?')}")
                for k, v in s.items():
                    if k != "policy":
                        print(f"    {k}: {v:.4f}" if isinstance(v, float)
                              else f"    {k}: {v}")
        return

    summaries = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        summaries.append(data.get("summary", data))

    # 表格输出
    metrics = [
        "successful", "failed",
        "ttft_mean", "ttft_median", "ttft_p90", "ttft_p95", "ttft_p99",
        "throughput_rps",
    ]

    col_w = 16
    header = f"{'指标':<20}"
    for s in summaries:
        header += f"  {s.get('policy', '?'):>{col_w}}"
    print(header)
    print("-" * len(header))

    for m in metrics:
        row = f"  {m:<18}"
        for s in summaries:
            v = s.get(m, "N/A")
            row += f"  {v:>{col_w}.4f}" if isinstance(v, float) else \
                   f"  {str(v):>{col_w}}"
        print(row)

    # 计算改进幅度
    lru_s = next((s for s in summaries if s.get("policy") == "lru"), None)
    od_s = next((s for s in summaries if s.get("policy") == "online_dual"), None)
    if lru_s and od_s:
        print(f"\n{'─'*50}")
        if lru_s.get("ttft_mean") and od_s.get("ttft_mean"):
            imp = (lru_s["ttft_mean"] - od_s["ttft_mean"]) / lru_s["ttft_mean"] * 100
            print(f"  TTFT 均值改进 (OnlineDual vs LRU): {imp:+.2f}%")
        if lru_s.get("ttft_p90") and od_s.get("ttft_p90"):
            imp = (lru_s["ttft_p90"] - od_s["ttft_p90"]) / lru_s["ttft_p90"] * 100
            print(f"  TTFT P90 改进 (OnlineDual vs LRU):  {imp:+.2f}%")
        if lru_s.get("throughput_rps") and od_s.get("throughput_rps"):
            imp = (od_s["throughput_rps"] - lru_s["throughput_rps"]) / lru_s["throughput_rps"] * 100
            print(f"  吞吐量改进 (OnlineDual vs LRU):     {imp:+.2f}%")


# ======================== 主入口 ========================
def main():
    parser = argparse.ArgumentParser(
        description="Disagg encoder 缓存策略压力测试"
    )

    # 模式
    parser.add_argument("--prepare-images", action="store_true",
                        help="生成合成图片数据库")
    parser.add_argument("--compare", action="store_true",
                        help="对比多次运行结果")

    # 服务
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--proxy-port", type=int, default=10001,
                        help="Disagg proxy 端口")

    # 图片
    parser.add_argument("--image-dir", type=str,
                        default="/tmp/vllm_disagg_bench_images")
    parser.add_argument("--num-images", type=int, default=30,
                        help="图片数据库大小")

    # 负载
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="请求总数")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="最大并发请求数 (默认 4)")
    parser.add_argument("--request-rate", type=float, default=2.0,
                        help="请求速率 (req/s)，0 表示尽快发送。默认 2.0")
    parser.add_argument("--distribution", type=str, default="zipf",
                        choices=["zipf", "uniform", "bimodal"])
    parser.add_argument("--zipf-alpha", type=float, default=1.5,
                        help="Zipf 分布参数，越大越集中")
    parser.add_argument("--seed", type=int, default=42)

    # 输出
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--policy-name", type=str, default="unknown")
    parser.add_argument("--results-dir", type=str,
                        default="./logs/disagg_cache_bench")
    parser.add_argument("--timestamp", type=str, default=None)

    args = parser.parse_args()

    # ---- 模式分发 ----
    if args.prepare_images:
        prepare_images(args.image_dir, args.num_images, args.seed)
        return

    if args.compare:
        compare_results(args.results_dir, args.timestamp)
        return

    # ---- 生成请求序列 ----
    image_indices = generate_request_sequence(
        num_images=args.num_images,
        num_prompts=args.num_prompts,
        distribution=args.distribution,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )

    counts = Counter(image_indices)
    print(f"\n请求分布 ({args.distribution}, alpha={args.zipf_alpha}):")
    print(f"  总请求数: {args.num_prompts}")
    print(f"  引用不同图片数: {len(counts)}/{args.num_images}")
    top5 = counts.most_common(5)
    print(f"  最热门 5 张: {[(f'img_{k:04d}', v) for k, v in top5]}")
    print()

    # ---- 发送请求 ----
    print(f"策略: {args.policy_name}")
    results = asyncio.run(
        run_workload(
            model=args.model,
            proxy_port=args.proxy_port,
            image_dir=args.image_dir,
            image_indices=image_indices,
            concurrency=args.concurrency,
            request_rate=args.request_rate,
        )
    )

    # ---- 分析结果 ----
    config = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "num_images": args.num_images,
        "distribution": args.distribution,
        "zipf_alpha": args.zipf_alpha,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "seed": args.seed,
    }
    summary = analyze(results, args.policy_name, config)

    print(f"\n{'='*50}")
    print(f"策略: {args.policy_name}")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- 保存 ----
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "config": config,
                        "raw_results": results}, f, indent=2)
        print(f"\n结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
