#!/bin/bash
# Full eval sweep: 4 suites × 3 seeds × {EM, NL}
# Skips already-completed runs (results/{suite}/KF_{EM|NL}_seed{N}/results.json present)
#
# Usage:
#   bash examples/LIBERO/run_eval_sweep.sh
# Resume-safe: kill anytime, re-run to continue from where it stopped.

set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

YOUR_CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
CKPT_ROOT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints
N_TRIALS=50
PORT=15090

SUITES=(libero_spatial libero_object libero_goal libero_10)
SEEDS=(7 42 123)
METHODS=(NL EM)

# Loop order: method → seed → suite (NL first, then EM)
# Reason: NL is the main contribution; finish all NL runs (12 total) before
# the EM ablation. Partial sweeps still give the headline NL result.
total_runs=$((${#SUITES[@]} * ${#SEEDS[@]} * ${#METHODS[@]}))
done_count=0; skip_count=0; run_count=0

for METHOD in "${METHODS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for SUITE in "${SUITES[@]}"; do
      done_count=$((done_count + 1))

      # Pick LDS path based on method
      if [ "$METHOD" = "EM" ]; then
        LDS=$CKPT_ROOT/${SUITE}_lds_tokens_em/lds_em_64.npz
      else
        LDS=$CKPT_ROOT/${SUITE}_nonlinear_lds/lds_nl_64.npz
      fi

      OUT="results/${SUITE}/KF_${METHOD}_seed${SEED}"
      mkdir -p "$OUT"

      # Skip if already done
      if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
        sr=$(grep "Total success rate" "$OUT/eval.log" | tail -1 | grep -oP "[0-9.]+(?= )" | head -1)
        echo "[$done_count/$total_runs] SKIP  $SUITE  $METHOD  seed=$SEED  (already done: SR=${sr})"
        skip_count=$((skip_count + 1))
        continue
      fi

      if [ ! -f "$LDS" ]; then
        echo "[$done_count/$total_runs] ERROR $SUITE  $METHOD  seed=$SEED  (no LDS at $LDS)"
        continue
      fi

      echo ""
      echo "════════════════════════════════════════════════════════════"
      echo "[$done_count/$total_runs] $SUITE  $METHOD  seed=$SEED"
      echo "  LDS: $LDS"
      echo "  Out: $OUT"
      echo "════════════════════════════════════════════════════════════"

      # Kill any leftover server on the port
      fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2

      # Start server (no --kf_q/--kf_r → uses EM-learned values from LDS)
      SERVER_LOG=/tmp/vla_server_sweep.log
      rm -f $SERVER_LOG
      python ./deployment/model_server/server_policy.py \
          --ckpt_path $YOUR_CKPT \
          --port $PORT \
          --use_bf16 \
          --cuda 0 \
          --lds_path $LDS > $SERVER_LOG 2>&1 &
      SERVER_PID=$!

      # Wait for server
      elapsed=0
      until grep -q "server listening" $SERVER_LOG 2>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ $elapsed -ge 180 ]; then
          echo "ERROR: server startup timeout"; cat $SERVER_LOG | tail -20
          kill $SERVER_PID 2>/dev/null; exit 1
        fi
      done
      echo "Server up. KF info:"
      grep "KF\|EKF" $SERVER_LOG | tail -2

      # Run eval client
      ${sim_python} ./examples/LIBERO/eval_libero.py \
          --args.pretrained-path $YOUR_CKPT \
          --args.host "127.0.0.1" \
          --args.port $PORT \
          --args.task-suite-name "$SUITE" \
          --args.num-trials-per-task $N_TRIALS \
          --args.video-out-path "$OUT" \
          --args.seed $SEED \
          --args.with_state "true" 2>&1 | tee "$OUT/eval.log"

      kill $SERVER_PID 2>/dev/null || true
      sleep 5
      run_count=$((run_count + 1))

      sr=$(grep "Total success rate" "$OUT/eval.log" | tail -1 | grep -oP "[0-9.]+" | head -1)
      echo "→ DONE: SR=$sr"
    done
  done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "SWEEP COMPLETE  ($run_count newly run, $skip_count skipped)"
echo "════════════════════════════════════════════════════════════"
