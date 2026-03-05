#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end benchmark comparing LRU vs OnlineDual encoder cache policies.
#
# This script:
# 1. Starts a vLLM server with a multimodal model
# 2. Runs benchmark_encoder_cache_serving.py with random image requests
# 3. Collects TTFT, throughput, and cache hit metrics
# 4. Repeats with each cache policy
#
# Prerequisites:
#   - A multimodal model (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   - GPU available
#   - Image database prepared (script will create one if not present)
#
# Usage:
#   bash benchmarks/benchmark_encoder_cache_e2e.sh
#   # Or with custom settings:
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct NUM_PROMPTS=200 \
#     bash benchmarks/benchmark_encoder_cache_e2e.sh

set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_IMAGES="${NUM_IMAGES:-20}"
IMAGE_DIR="${IMAGE_DIR:-/tmp/vllm_bench_images}"
LOG_DIR="${LOG_DIR:-./logs/cache_benchmark}"
ZIPF_ALPHA="${ZIPF_ALPHA:-1.2}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")

mkdir -p "$LOG_DIR"
mkdir -p "$IMAGE_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

###############################################################################
# Helpers
###############################################################################
wait_for_server() {
    local port=$1
    echo "Waiting for server on port $port..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/health > /dev/null 2>&1; do
            sleep 2
        done" && echo "Server ready!" || { echo "Server failed to start"; return 1; }
}

cleanup() {
    echo "Cleaning up..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

###############################################################################
# Step 1: Prepare image database
###############################################################################
echo "=== Step 1: Preparing image database ==="
python3 "${GIT_ROOT}/benchmarks/benchmark_encoder_cache_serving.py" \
    --prepare-images \
    --image-dir "$IMAGE_DIR" \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED"

###############################################################################
# Step 2: Run benchmarks for each policy
###############################################################################
for POLICY in lru online_dual; do
    echo ""
    echo "============================================================"
    echo "  Running benchmark with policy: $POLICY"
    echo "============================================================"

    SERVER_LOG="${LOG_DIR}/server_${POLICY}_${TIMESTAMP}.log"
    RESULT_FILE="${LOG_DIR}/results_${POLICY}_${TIMESTAMP}.json"

    # Start server with specified cache policy
    echo "Starting vLLM server with $POLICY cache policy..."
    VLLM_CACHE_POLICY="$POLICY" vllm serve "$MODEL" \
        --port "$PORT" \
        --enforce-eager \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.8 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server "$PORT"; then
        echo "Failed to start server with $POLICY policy. Check $SERVER_LOG"
        kill "$SERVER_PID" 2>/dev/null || true
        continue
    fi

    # Run the benchmark
    echo "Running benchmark..."
    python3 "${GIT_ROOT}/benchmarks/benchmark_encoder_cache_serving.py" \
        --model "$MODEL" \
        --port "$PORT" \
        --image-dir "$IMAGE_DIR" \
        --num-prompts "$NUM_PROMPTS" \
        --num-images "$NUM_IMAGES" \
        --zipf-alpha "$ZIPF_ALPHA" \
        --seed "$SEED" \
        --output-json "$RESULT_FILE" \
        --policy-name "$POLICY"

    echo "Results saved to $RESULT_FILE"

    # Stop server
    echo "Stopping server..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
    sleep 3
done

###############################################################################
# Step 3: Compare results
###############################################################################
echo ""
echo "============================================================"
echo "  Comparison Summary"
echo "============================================================"
python3 "${GIT_ROOT}/benchmarks/benchmark_encoder_cache_serving.py" \
    --compare \
    --results-dir "$LOG_DIR" \
    --timestamp "$TIMESTAMP"

echo ""
echo "Benchmark complete. Results in $LOG_DIR"
