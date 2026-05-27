#!/bin/bash
# Sweep adaptive-R (per-step token-spread KF) on libero_10: 3 seeds × gamma values.
# Compares to EM KF (constant R) baseline: 96.40 ± 0.72 on libero_10.
#
# Usage: bash examples/LIBERO/sweep_adaptr.sh
# Resume-safe.

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
SUITE=libero_10
LDS=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/${SUITE}_lds_tokens_em/lds_em_64.npz
PORT=15095

GAMMAS=(1.0 2.0)      # spread sensitivity
SEEDS=(7 42 123)

total=$((${#GAMMAS[@]} * ${#SEEDS[@]})); done=0
for G in "${GAMMAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    done=$((done+1))
    OUT="results/${SUITE}/KF_ADAPTR_g${G}_seed${SEED}"
    if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
      sr=$(grep "Total success rate" "$OUT/eval.log" | tail -1 | grep -oP "[0-9.]+" | head -1)
      echo "[$done/$total] SKIP gamma=$G seed=$SEED (SR=$sr)"; continue
    fi
    mkdir -p "$OUT"
    echo ""; echo "════ [$done/$total] adaptive-R gamma=$G seed=$SEED ════"
    fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
    rm -f /tmp/vla_server_adaptr_sweep.log
    python ./deployment/model_server/server_policy.py \
        --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
        --lds_path $LDS --adaptive_r --ar_gamma $G > /tmp/vla_server_adaptr_sweep.log 2>&1 &
    SPID=$!
    el=0
    until grep -q "server listening" /tmp/vla_server_adaptr_sweep.log 2>/dev/null; do
      sleep 2; el=$((el+2))
      if [ $el -ge 180 ]; then echo "timeout"; cat /tmp/vla_server_adaptr_sweep.log|tail; kill $SPID; exit 1; fi
    done
    grep "Adaptive-R" /tmp/vla_server_adaptr_sweep.log
    $sim_python ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
        --args.task-suite-name "$SUITE" --args.num-trials-per-task 50 \
        --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
        2>&1 | tee /tmp/vla_adaptr_live.log | grep "Total success rate" | tail
    kill $SPID 2>/dev/null || true; sleep 5
  done
done

echo ""; echo "════ adaptive-R sweep done ════"
echo "vs EM KF (constant R) libero_10: s7=0.956 s42=0.970 s123=0.966 (mean 96.40)"
for G in "${GAMMAS[@]}"; do
  vals=""
  for SEED in "${SEEDS[@]}"; do
    sr=$(grep "Total success rate" "results/${SUITE}/KF_ADAPTR_g${G}_seed${SEED}/eval.log" 2>/dev/null | tail -1 | grep -oP "[0-9.]+" | head -1)
    vals="$vals s$SEED=${sr:-?}"
  done
  echo "  gamma=$G :$vals"
done
