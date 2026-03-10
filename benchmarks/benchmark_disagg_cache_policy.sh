#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Disaggregated Encoder 架构下的缓存策略对比测试
#
# 在 E+P+D 三节点架构中，用 Zipf 分布引用图片发请求，
# 分别测试 LRU 和 OnlineDual 缓存策略，对比 TTFT、吞吐量等指标。
#
# 需要 3 张 GPU（encoder / prefill / decode 各 1 张）
#
# 用法:
#   bash benchmarks/benchmark_disagg_cache_policy.sh
#
# 可选配置:
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
#   NUM_IMAGES=30 NUM_PROMPTS=200 ZIPF_ALPHA=1.5 \
#   GPU_E=0 GPU_P=1 GPU_D=2 \
#   bash benchmarks/benchmark_disagg_cache_policy.sh

set -euo pipefail

###############################################################################
# 配置
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs/disagg_cache_bench}"
IMAGE_DIR="${IMAGE_DIR:-/tmp/vllm_disagg_bench_images}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache_bench}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

NUM_IMAGES="${NUM_IMAGES:-30}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
ZIPF_ALPHA="${ZIPF_ALPHA:-1.5}"
CONCURRENCY="${CONCURRENCY:-4}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"
SEED="${SEED:-42}"

# Encoder cache size override (tokens). 0 = use vLLM default.
# Set this smaller than the total encoder output to create cache pressure.
# Example: 30 images × ~1000 tokens each ≈ 30000; use 5000-10000 for pressure.
ENCODER_CACHE_SIZE="${ENCODER_CACHE_SIZE:-5000}"

export UCX_TLS=all
export UCX_NET_DEVICES=all

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$LOG_PATH"
mkdir -p "$IMAGE_DIR"

declare -a PIDS=()

###############################################################################
# 工具函数
###############################################################################
wait_for_server() {
    local port=$1
    local name=${2:-"server"}
    echo "[INFO] 等待 ${name} 启动 (port=$port)..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/health > /dev/null 2>&1; do
            sleep 2
        done" && echo "[OK] ${name} 已就绪" || {
        echo "[ERROR] ${name} 启动超时"; return 1;
    }
}

cleanup() {
    echo "[INFO] 清理进程..."
    trap - INT TERM USR1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    echo "[INFO] 清理完成"
}
trap cleanup EXIT INT TERM

start_disagg_cluster() {
    local policy=$1
    local log_prefix="${LOG_PATH}/${policy}_${TIMESTAMP}"

    echo "[INFO] 清理旧缓存..."
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    PIDS=()

    # ---- Encoder Worker ----
    echo "[INFO] 启动 Encoder (GPU=$GPU_E, policy=$policy, cache_size=$ENCODER_CACHE_SIZE)..."
    VLLM_CACHE_POLICY="$policy" \
    VLLM_ENCODER_CACHE_SIZE="$ENCODER_CACHE_SIZE" \
    CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
        --gpu-memory-utilization 0.01 \
        --port "$ENCODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        >"${log_prefix}_encoder.log" 2>&1 &
    PIDS+=($!)

    # ---- Prefill Worker ----
    echo "[INFO] 启动 Prefill (GPU=$GPU_P, policy=$policy)..."
    VLLM_CACHE_POLICY="$policy" \
    CUDA_VISIBLE_DEVICES="$GPU_P" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    vllm serve "$MODEL" \
        --gpu-memory-utilization 0.7 \
        --port "$PREFILL_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer"
        }' \
        >"${log_prefix}_prefill.log" 2>&1 &
    PIDS+=($!)

    # ---- Decode Worker ----
    echo "[INFO] 启动 Decode (GPU=$GPU_D)..."
    CUDA_VISIBLE_DEVICES="$GPU_D" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
    vllm serve "$MODEL" \
        --gpu-memory-utilization 0.7 \
        --port "$DECODE_PORT" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path "$IMAGE_DIR" \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer"
        }' \
        >"${log_prefix}_decode.log" 2>&1 &
    PIDS+=($!)

    # 等待所有 worker 就绪
    wait_for_server $ENCODE_PORT "Encoder"
    wait_for_server $PREFILL_PORT "Prefill"
    wait_for_server $DECODE_PORT "Decode"

    # ---- Proxy ----
    echo "[INFO] 启动 Proxy..."
    python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
        --host "0.0.0.0" \
        --port "$PROXY_PORT" \
        --encode-servers-urls "http://localhost:$ENCODE_PORT" \
        --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
        --decode-servers-urls "http://localhost:$DECODE_PORT" \
        >"${log_prefix}_proxy.log" 2>&1 &
    PIDS+=($!)

    wait_for_server $PROXY_PORT "Proxy"
    echo "[OK] 所有服务已启动 (policy=$policy)"
}

stop_disagg_cluster() {
    echo "[INFO] 停止集群..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    PIDS=()
    sleep 2
}

###############################################################################
# Step 1: 准备图片数据库
###############################################################################
echo ""
echo "============================================================"
echo "  Step 1: 准备图片数据库 (${NUM_IMAGES} 张图片)"
echo "============================================================"

python3 "${GIT_ROOT}/benchmarks/benchmark_disagg_cache_workload.py" \
    --prepare-images \
    --image-dir "$IMAGE_DIR" \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED"

###############################################################################
# Step 2: 分别测试两种策略
###############################################################################
for POLICY in lru online_dual; do
    echo ""
    echo "============================================================"
    echo "  Step 2: 测试策略 $POLICY"
    echo "============================================================"

    RESULT_FILE="${LOG_PATH}/results_${POLICY}_${TIMESTAMP}.json"

    start_disagg_cluster "$POLICY"

    echo "[INFO] 发送 ${NUM_PROMPTS} 个请求 (Zipf alpha=${ZIPF_ALPHA})..."
    python3 "${GIT_ROOT}/benchmarks/benchmark_disagg_cache_workload.py" \
        --model "$MODEL" \
        --proxy-port "$PROXY_PORT" \
        --image-dir "$IMAGE_DIR" \
        --num-images "$NUM_IMAGES" \
        --num-prompts "$NUM_PROMPTS" \
        --zipf-alpha "$ZIPF_ALPHA" \
        --concurrency "$CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --seed "$SEED" \
        --policy-name "$POLICY" \
        --output-json "$RESULT_FILE"

    echo "[INFO] 结果已保存: $RESULT_FILE"

    stop_disagg_cluster
done

###############################################################################
# Step 3: 对比结果
###############################################################################
echo ""
echo "============================================================"
echo "  Step 3: 对比结果"
echo "============================================================"

python3 "${GIT_ROOT}/benchmarks/benchmark_disagg_cache_workload.py" \
    --compare \
    --results-dir "$LOG_PATH" \
    --timestamp "$TIMESTAMP"

echo ""
echo "[DONE] 完整结果在 $LOG_PATH"
