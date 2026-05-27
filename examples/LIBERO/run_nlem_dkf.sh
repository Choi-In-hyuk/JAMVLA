#!/bin/bash
# Master pipeline: train + eval NL+EM (EKF-EM) and DKF across 4 suites × 3 seeds.
#
# Stage A (CPU, ~1h): train NL+EM and DKF checkpoints for all 4 suites
# Stage B (GPU, ~24h): eval sweep — NL+EM ×12 then DKF ×12 runs
#
# Usage:  bash examples/LIBERO/run_nlem_dkf.sh
# Resume-safe: re-run to continue; completed evals are skipped.

set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
_NVIDIA_LD=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

VJEPA=/home/choi/vjepa2
DATA=/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets
CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints
YOUR_CKPT=$CKPT/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
PY=/home/choi/miniconda3/envs/vla_jepa/bin/python

SUITES=(libero_spatial libero_object libero_goal libero_10)
SEEDS=(7 42 123)
PORT=15091
N_TRIALS=50

# ═══════════════════════════════════════════════════════════════════
# STAGE A: Training (CPU — no GPU contention with eval)
# ═══════════════════════════════════════════════════════════════════
echo "════════ STAGE A: Training NL+EM and DKF checkpoints ════════"
cd $VJEPA
for SUITE in "${SUITES[@]}"; do
  TOK=$DATA/$SUITE/tokens_vla_jepa
  NLEM_DIR=$CKPT/${SUITE}_nlem_lds
  DKF_DIR=$CKPT/${SUITE}_dkf

  # NL + EKF-EM
  if [ -f "$NLEM_DIR/lds_nl_64_mlp.pt" ]; then
    echo "[$SUITE] NL+EM exists, skip"
  else
    echo "──── Training NL+EM for $SUITE ────"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    LD_LIBRARY_PATH=$_NVIDIA_LD:$LD_LIBRARY_PATH \
    $PY -u -m experiments.vla_jepa.train_nonlinear_lds \
        --token_dir $TOK --ckpt_dir $NLEM_DIR \
        --epochs 150 --use_ekf_em --em_iter 20 --em_max_seqs 150 \
        --device cpu 2>&1 | tail -6
  fi

  # DKF
  if [ -f "$DKF_DIR/dkf_64_nets.pt" ]; then
    echo "[$SUITE] DKF exists, skip"
  else
    echo "──── Training DKF for $SUITE ────"
    OMP_NUM_THREADS=4 \
    LD_LIBRARY_PATH=$_NVIDIA_LD:$LD_LIBRARY_PATH \
    $PY -u -m experiments.vla_jepa.train_token_dkf \
        --token_dir $TOK --ckpt_dir $DKF_DIR \
        --epochs 80 --kl_weight 3.0 --device cpu 2>&1 | tail -6
  fi
done

echo ""
echo "════════ STAGE A done. Checkpoints: ════════"
for SUITE in "${SUITES[@]}"; do
  nlem=$([ -f $CKPT/${SUITE}_nlem_lds/lds_nl_64_mlp.pt ] && echo ✓ || echo ✗)
  dkf=$([ -f $CKPT/${SUITE}_dkf/dkf_64_nets.pt ] && echo ✓ || echo ✗)
  echo "  $SUITE  NL+EM=$nlem  DKF=$dkf"
done

# ═══════════════════════════════════════════════════════════════════
# STAGE B: Eval sweep (GPU)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "════════ STAGE B: Eval sweep (NL+EM ×12, then DKF ×12) ════════"
cd /home/choi/VLA-JEPA
export LD_LIBRARY_PATH=$_NVIDIA_LD$LD_LIBRARY_PATH

run_eval () {
  local METHOD=$1 SUITE=$2 SEED=$3 LDS=$4
  local OUT="results/${SUITE}/KF_${METHOD}_seed${SEED}"
  mkdir -p "$OUT"
  if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
    local sr=$(grep "Total success rate" "$OUT/eval.log" | tail -1 | grep -oP "[0-9.]+" | head -1)
    echo "  SKIP $METHOD $SUITE seed=$SEED (SR=$sr)"; return
  fi
  if [ ! -f "$LDS" ]; then echo "  ERROR no ckpt: $LDS"; return; fi

  echo "──── $METHOD  $SUITE  seed=$SEED ────"
  fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2
  rm -f /tmp/vla_server_nlemdkf.log
  $PY ./deployment/model_server/server_policy.py \
      --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
      --lds_path $LDS > /tmp/vla_server_nlemdkf.log 2>&1 &
  local SPID=$!
  local el=0
  until grep -q "server listening" /tmp/vla_server_nlemdkf.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 180 ]; then echo "  server timeout"; cat /tmp/vla_server_nlemdkf.log|tail -10; kill $SPID 2>/dev/null; return; fi
  done
  grep -E "\[KF\]|\[EKF\]|\[DKF\]" /tmp/vla_server_nlemdkf.log | tail -1
  $sim_python ./examples/LIBERO/eval_libero.py \
      --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
      --args.task-suite-name "$SUITE" --args.num-trials-per-task $N_TRIALS \
      --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
      2>&1 | tee "$OUT/eval.log" | grep -E "Total success rate" | tail -1
  kill $SPID 2>/dev/null || true; sleep 5
}

# NL+EM first (all suites/seeds), then DKF
for SEED in "${SEEDS[@]}"; do
  for SUITE in "${SUITES[@]}"; do
    run_eval "NLEM" "$SUITE" "$SEED" "$CKPT/${SUITE}_nlem_lds/lds_nl_64.npz"
  done
done
for SEED in "${SEEDS[@]}"; do
  for SUITE in "${SUITES[@]}"; do
    run_eval "DKF" "$SUITE" "$SEED" "$CKPT/${SUITE}_dkf/dkf_64.npz"
  done
done

echo ""
echo "════════ ALL DONE ════════"
python3 scripts/make_sweep_report.py 2>/dev/null || true
