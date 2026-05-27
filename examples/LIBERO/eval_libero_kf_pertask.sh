#!/bin/bash
# Per-task EM KF eval — VLA-JEPA on libero_10, each task uses its own EM LDS.
# Server auto-selects the task-specific LDS from the instruction string.
#
# Usage: bash eval_libero_kf_pertask.sh [task_suite] [seed]

set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

YOUR_CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
SUITE=${1:-libero_10}
SEED=${2:-7}
PERTASK_DIR=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/${SUITE}_pertask_lds_em
PORT=15094

OUT="results/${SUITE}/KF_PERTASK_seed${SEED}"
mkdir -p "$OUT"

if [ ! -f "$PERTASK_DIR/task_index.json" ]; then
    echo "ERROR: per-task LDS not found at $PERTASK_DIR"
    echo "Train first: python -m experiments.vla_jepa.train_per_task_lds_em ..."
    exit 1
fi

fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
rm -f /tmp/vla_server_pertask.log
python ./deployment/model_server/server_policy.py \
    --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
    --per_task_dir $PERTASK_DIR > /tmp/vla_server_pertask.log 2>&1 &
SPID=$!
echo "Server PID: $SPID"

el=0
until grep -q "server listening" /tmp/vla_server_pertask.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 180 ]; then echo "timeout"; cat /tmp/vla_server_pertask.log | tail -10; kill $SPID; exit 1; fi
done
grep "Per-task KF" /tmp/vla_server_pertask.log

# Live logging to /tmp/vla_pertask_live.log  (watch:  tail -f /tmp/vla_pertask_live.log)
$sim_python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
    --args.task-suite-name "$SUITE" --args.num-trials-per-task 50 \
    --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
    2>&1 | tee /tmp/vla_pertask_live.log | grep -E "success rate|switched to"

kill $SPID 2>/dev/null || true
echo "Done. Output: $OUT"
