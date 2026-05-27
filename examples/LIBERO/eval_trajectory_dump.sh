#!/bin/bash
# Re-run baseline + EM KF on libero_10 seed 7 with per-step trajectory + final
# object positions saved. For paper figure: rollout vs demo GT trajectory.
#
# Output:
#   results/libero_10/baseline_traj_seed7/{eval.log, final_positions.jsonl, trajectories/*.npz}
#   results/libero_10/KF_EM_traj_seed7/    {same}

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
SEED=7
LDS=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/${SUITE}_lds_tokens_em/lds_em_64.npz
PORT=15098

run_eval(){
  local NAME=$1 USE_KF=$2
  local OUT="results/${SUITE}/${NAME}"
  if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
    echo "[SKIP] $NAME (already done)"; return
  fi
  mkdir -p "$OUT"; rm -f "$OUT/final_positions.jsonl"
  echo ""; echo "════ $NAME ════"
  fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
  rm -f /tmp/vla_server_traj.log
  if [ "$USE_KF" = "1" ]; then
    python ./deployment/model_server/server_policy.py \
        --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
        --lds_path $LDS > /tmp/vla_server_traj.log 2>&1 &
  else
    python ./deployment/model_server/server_policy.py \
        --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
        > /tmp/vla_server_traj.log 2>&1 &
  fi
  SPID=$!
  el=0
  until grep -q "server listening" /tmp/vla_server_traj.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 180 ]; then echo "timeout"; cat /tmp/vla_server_traj.log|tail; kill $SPID; exit 1; fi
  done
  grep -E "KF enabled|server listening" /tmp/vla_server_traj.log | tail -2

  $sim_python ./examples/LIBERO/eval_libero.py \
      --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
      --args.task-suite-name "$SUITE" --args.num-trials-per-task 50 \
      --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
      --args.log-positions --args.log-trajectory \
      2>&1 | tee /tmp/vla_traj_live.log | grep -E "Total success rate" | tail
  kill $SPID 2>/dev/null || true; sleep 5
}

run_eval baseline_traj_seed7 0
run_eval KF_EM_traj_seed7    1

echo ""; echo "════ done ════"
for d in baseline_traj_seed7 KF_EM_traj_seed7; do
  sr=$(grep "Total success rate" "results/${SUITE}/${d}/eval.log" 2>/dev/null | tail -1 | grep -oP "[0-9.]+" | head -1)
  n=$(ls "results/${SUITE}/${d}/trajectories"/*.npz 2>/dev/null | wc -l)
  echo "  $d  SR=$sr  trajectories=$n"
done
