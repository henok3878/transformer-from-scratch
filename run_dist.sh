#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
    set -a 
    source .env 
    set +a 
fi

HOSTNAME=$(hostname)
if [ "$HOSTNAME" = "$MASTER_ADDR" ]; then
  NODE_RANK=0
else
  NODE_RANK=1
fi

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
