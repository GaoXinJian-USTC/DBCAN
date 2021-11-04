#!/usr/bin/env bash

# if [ $# -lt 3 ]
# then
#     echo "Usage: bash $0 CONFIG CHECKPOINT GPUS"
#     exit
# fi

CONFIG=configs/DBCAN.py
CHECKPOINT=best.pth
GPUS=2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch  --eval acc ${@:4}

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 tools/test.py roubust_adaptive_encoding/test_config.py roubust_adaptive_encoding/epoch_6.pth --launcher pytorch  --eval acc 