#!/bin/bash
# Fresh 3-seed reproduction: baseline + EM KF on libero_10, seeds 7/42/123.
# Goal: re-establish the 3-seed-mean baseline-vs-KF comparison (results.md §3.1)
# and check whether EM KF's advantage is reproduced or was single-seed noise.
#
# Output: results/libero_10/{base_3s_seed7,kf_em_3s_seed7,...}/eval.log
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$(pwd):$PYTHONPATH:${LIBERO_HOME}
sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

YOUR_CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
SUITE=libero_10
LDS=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/${SUITE}_lds_tokens_em/lds_em_64.npz
PORT=15098

run_eval(){
  local NAME=$1 USE_KF=$2 SEED=$3
  local OUT="results/${SUITE}/${NAME}"
  if [ -f "$OUT/eval.log" ] && grep -q "Total success rate" "$OUT/eval.log" 2>/dev/null; then
    echo "[SKIP] $NAME (already done: $(grep 'Total success rate' "$OUT/eval.log"|tail -1|grep -oE '[0-9.]+'|head -1))"; return
  fi
  mkdir -p "$OUT"
  echo ""; echo "════ $NAME (KF=$USE_KF seed=$SEED) ════"
  fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
  rm -f /tmp/vla_server_3s.log
  if [ "$USE_KF" = "1" ]; then
    python ./deployment/model_server/server_policy.py --ckpt_path $YOUR_CKPT \
        --port $PORT --use_bf16 --cuda 0 --lds_path $LDS > /tmp/vla_server_3s.log 2>&1 &
  else
    python ./deployment/model_server/server_policy.py --ckpt_path $YOUR_CKPT \
        --port $PORT --use_bf16 --cuda 0 > /tmp/vla_server_3s.log 2>&1 &
  fi
  SPID=$!
  el=0
  until grep -q "server listening" /tmp/vla_server_3s.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 240 ]; then echo "server timeout"; tail /tmp/vla_server_3s.log; kill $SPID; exit 1; fi
  done
  $sim_python ./examples/LIBERO/eval_libero.py \
      --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
      --args.task-suite-name "$SUITE" --args.num-trials-per-task 50 \
      --args.video-out-path "$OUT" --args.seed $SEED --args.with_state "true" \
      2>&1 | tee "$OUT/eval.log" | grep -E "Total success rate" | tail
  kill $SPID 2>/dev/null || true; sleep 5
}

for SEED in 7 42 123; do
  run_eval base_3s_seed${SEED}  0 $SEED
  run_eval kf_em_3s_seed${SEED} 1 $SEED
done

echo ""; echo "════ DONE — 3-seed summary ════"
for m in base kf_em; do
  for SEED in 7 42 123; do
    f="results/${SUITE}/${m}_3s_seed${SEED}/eval.log"
    sr=$(grep "Total success rate" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1)
    echo "  ${m}_seed${SEED}: ${sr}"
  done
done
