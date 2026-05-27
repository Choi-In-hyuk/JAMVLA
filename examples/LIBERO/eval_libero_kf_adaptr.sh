#!/bin/bash
# EM KF + adaptive observation noise R_t (from token spread) — true adaptive KF.
# Usage: bash eval_libero_kf_adaptr.sh [task_suite] [seed] [gamma]

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
GAMMA=${3:-1.0}
LDS=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/${SUITE}_lds_tokens_em/lds_em_64.npz
PORT=15095

OUT="results/${SUITE}/KF_ADAPTR_g${GAMMA}_seed${SEED}"
mkdir -p "$OUT"

fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
rm -f /tmp/vla_server_adaptr.log
python ./deployment/model_server/server_policy.py \
    --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
    --lds_path $LDS \
    --adaptive_r --ar_gamma $GAMMA > /tmp/vla_server_adaptr.log 2>&1 &
SPID=$!
echo "Server PID: $SPID"
el=0
until grep -q "server listening" /tmp/vla_server_adaptr.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 180 ]; then echo "timeout"; cat /tmp/vla_server_adaptr.log|tail; kill $SPID; exit 1; fi
done
grep -E "KF enabled|Adaptive-R" /tmp/vla_server_adaptr.log

# Live: tail -f /tmp/vla_adaptr_live.log
$sim_python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
    --args.task-suite-name "$SUITE" --args.num-trials-per-task 50 \
    --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
    2>&1 | tee /tmp/vla_adaptr_live.log | grep -E "success rate"

kill $SPID 2>/dev/null || true
echo "Done. Output: $OUT"
