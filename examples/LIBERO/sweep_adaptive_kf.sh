#!/bin/bash
# Sweep adaptive KF on libero_10: 3 seeds × 3 placement_alpha values = 9 runs.
# Compared to EM KF baseline (95.6 at seed 7, 96.4 average).
#
# Usage: bash examples/LIBERO/sweep_adaptive_kf.sh
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
PORT=15093
N_TRIALS=50

ALPHAS=(0.0 0.2 0.5)
SEEDS=(7 42 123)

total=9; done=0; ran=0; skipped=0
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    done=$((done+1))
    OUT="results/${SUITE}/KF_EMADAPT_a${ALPHA}_seed${SEED}"
    if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
      sr=$(grep "Total success rate" "$OUT/eval.log" | tail -1 | grep -oP "[0-9.]+" | head -1)
      echo "[$done/$total] SKIP alpha=$ALPHA seed=$SEED (SR=$sr)"
      skipped=$((skipped+1))
      continue
    fi
    mkdir -p "$OUT"

    echo ""
    echo "════════════════════════════════════════"
    echo "[$done/$total] alpha=$ALPHA  seed=$SEED"
    echo "════════════════════════════════════════"

    fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
    rm -f /tmp/vla_server_adapt_sweep.log
    python ./deployment/model_server/server_policy.py \
        --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
        --lds_path $LDS \
        --adaptive --placement_alpha $ALPHA > /tmp/vla_server_adapt_sweep.log 2>&1 &
    SPID=$!
    el=0
    until grep -q "server listening" /tmp/vla_server_adapt_sweep.log 2>/dev/null; do
      sleep 2; el=$((el+2))
      if [ $el -ge 180 ]; then echo "timeout"; cat /tmp/vla_server_adapt_sweep.log | tail -10; kill $SPID; exit 1; fi
    done
    grep -E "KF enabled|Adaptive" /tmp/vla_server_adapt_sweep.log

    $sim_python ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
        --args.task-suite-name "$SUITE" --args.num-trials-per-task $N_TRIALS \
        --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
        --args.log-positions \
        2>&1 | tee "$OUT/eval.log" | grep "Total success rate" | tail
    kill $SPID 2>/dev/null || true; sleep 5
    ran=$((ran+1))
  done
done

echo ""
echo "════════ SWEEP DONE  (ran=$ran  skipped=$skipped) ════════"
echo "Comparison vs EM KF baseline:"
for ALPHA in "${ALPHAS[@]}"; do
  vals=""
  for SEED in "${SEEDS[@]}"; do
    sr=$(grep "Total success rate" "results/${SUITE}/KF_EMADAPT_a${ALPHA}_seed${SEED}/eval.log" 2>/dev/null | tail -1 | grep -oP "[0-9.]+" | head -1)
    vals="$vals s$SEED=${sr:-?}"
  done
  echo "  alpha=$ALPHA :$vals"
done
echo "  EM KF      : s7=0.956  s42=0.970  s123=0.966  (baseline)"
