#!/usr/bin/env bash

# if [ $# -lt 3 ]
# then
#     echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
#     exit
# fi

CONFIG=configs/DBCAN.py
WORK_DIR=./roubust_two_stage/
GPUS=2

PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} ${@:4}
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR}   --launcher pytorch ${@:4}
fi

