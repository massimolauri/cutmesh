#!/usr/bin/env bash
# =============================================================================
# Cutmesh P2P worker — zero-config entrypoint.
#
# `docker run --gpus all -p 40000:40000/udp <image>` must be enough:
#   - unique, persistent node identity (no --seed)
#   - P2P: relay used only for signaling/hole punching, tensor data always on a direct path
#   - public ip:port auto-detected (CUTMESH_PUBLIC_ADDR > Vast.ai env > HTTP probe),
#     UPnP/NAT-PMP port mapping attempted automatically on home routers
#
# Passing arguments runs the binary directly, e.g. `<image> gateway --model ...`.
# =============================================================================
set -euo pipefail

BIN=/usr/local/bin/cutmesh

if [ "$#" -gt 0 ]; then
  exec "$BIN" "$@"
fi

export CUTMESH_DATA_DIR="${CUTMESH_DATA_DIR:-${SYNAPSE_DATA_DIR:-/data/cutmesh}}"
export LLAMA_CACHE="${LLAMA_CACHE:-${CUTMESH_DATA_DIR}/cache}"
export CUTMESH_HUB_URL="${CUTMESH_HUB_URL:-${SYNAPSE_HUB_URL:-}}"
export CUTMESH_NODE_TOKEN="${CUTMESH_NODE_TOKEN:-${SYNAPSE_NODE_TOKEN:-}}"
export CUTMESH_NODE_NAME="${CUTMESH_NODE_NAME:-${SYNAPSE_NODE_NAME:-}}"
export CUTMESH_POOL_MODE="${CUTMESH_POOL_MODE:-${SYNAPSE_POOL_MODE:-}}"
export CUTMESH_POWER_PCT="${CUTMESH_POWER_PCT:-${SYNAPSE_POWER_PCT:-100}}"
export CUTMESH_STRICT_DIRECT="${CUTMESH_STRICT_DIRECT:-${SYNAPSE_STRICT_DIRECT:-1}}"
export CUTMESH_QUIC_PORT="${CUTMESH_QUIC_PORT:-${SYNAPSE_QUIC_PORT:-40000}}"
export CUTMESH_PUBLIC_ADDR="${CUTMESH_PUBLIC_ADDR:-${SYNAPSE_PUBLIC_ADDR:-}}"
mkdir -p "$CUTMESH_DATA_DIR" "${LLAMA_CACHE}/rpc"

PREWARM_MODEL="${CUTMESH_PREWARM_MODEL:-${SYNAPSE_PREWARM_MODEL:-}}"
if [ -n "$PREWARM_MODEL" ]; then
  echo "[cutmesh] pre-downloading weights from ${PREWARM_MODEL} into ${LLAMA_CACHE}/rpc ..."
  "$BIN" pull-rig-shard \
    --model "$PREWARM_MODEL" \
    --cache-dir "${LLAMA_CACHE}/rpc" \
    --start-layer "${CUTMESH_START_LAYER:-${SYNAPSE_START_LAYER:-0}}" \
    --end-layer "${CUTMESH_END_LAYER:-${SYNAPSE_END_LAYER:-999}}" \
    --include-non-block || echo "[cutmesh] WARN: initial pre-download failed, will retry on gateway request"
fi

QUIC_PORT="${CUTMESH_QUIC_PORT:-40000}"
RPC_PORT="${CUTMESH_RPC_PORT:-${SYNAPSE_RPC_PORT:-50052}}"
DEVICE="${CUTMESH_CUDA_DEVICE:-${SYNAPSE_CUDA_DEVICE:-CUDA0}}"
LLAMA_BIN_DIR="${CUTMESH_LLAMA_BIN_DIR:-${SYNAPSE_LLAMA_BIN_DIR:-/opt/cutmesh/llamacpp}}"

if [ -z "${CUTMESH_NODE_NAME:-}" ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/NVIDIA //; s/GeForce //; s/ /-/g' || true)"
  CUTMESH_NODE_NAME="${GPU_NAME:-gpu}-$(hostname | cut -c1-8)"
fi

echo "[cutmesh] worker '${CUTMESH_NODE_NAME}' | QUIC udp/${QUIC_PORT} | ggml-rpc tcp/127.0.0.1:${RPC_PORT} | device ${DEVICE}"
if [ -n "${CUTMESH_PUBLIC_ADDR:-}" ]; then
  echo "[cutmesh] explicit public address: ${CUTMESH_PUBLIC_ADDR}"
elif [ -n "${PUBLIC_IPADDR:-}" ]; then
  VAR="VAST_UDP_PORT_${QUIC_PORT}"
  echo "[cutmesh] Vast.ai: ${PUBLIC_IPADDR}:${!VAR:-<UDP port ${QUIC_PORT} NOT exposed: add -p ${QUIC_PORT}:${QUIC_PORT}/udp>}"
fi

exec "$BIN" worker \
  --name "$CUTMESH_NODE_NAME" \
  --quic-port "$QUIC_PORT" \
  --local-rpc-port "$RPC_PORT" \
  --device "$DEVICE" \
  --llama-bin-dir "$LLAMA_BIN_DIR" \
  --wan-dht
