#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
    set -a 
    source .env 
    set +a 
fi

HOSTNAME=$(hostname)

# check for hostfile
if [[ ! -f hostfile ]]; then
    echo "Error: hostfile not found in current directory."
    exit 1
fi

# find NODE_RANK from hostfile
NODE_RANK=$(grep -nxw "$HOSTNAME" hostfile | cut -d: -f1 || true)
if [[ -z "$NODE_RANK" ]]; then
    echo "Error: $HOSTNAME not found in hostfile."
    exit 1
fi
NODE_RANK=$((NODE_RANK - 1))

echo "Launching distributed training:"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NNODES=${NNODES}"
echo "  GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  NCCL_DEBUG=${NCCL_DEBUG}"

# launch 
torchrun \
  --nnodes ${NNODES} \
  --nproc_per_node ${GPUS_PER_NODE} \
  --rdzv_id transformer_run \
  --rdzv_backend c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  --node_rank ${NODE_RANK} \
  train.py \
    --config ./configs/config_de-en.yaml
