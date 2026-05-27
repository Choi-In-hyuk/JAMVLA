#!/bin/bash
# Re-run EM KF on libero_10 task 4 (white+yellow mug) with position logging.
# Then analyze the final mug ↔ plate distances to test "subtle precision degradation".
#
# Usage: bash examples/LIBERO/analyze_em_kf_positions.sh

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
LDS=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_10_lds_tokens_em/lds_em_64.npz
PORT=15092
OUT=results/libero_10/EM_KF_task4_positions_seed7
mkdir -p $OUT
rm -f $OUT/final_positions.jsonl

# Start EM KF server
fuser -k $PORT/tcp 2>/dev/null || true; sleep 2
rm -f /tmp/vla_server_posanalysis.log
python ./deployment/model_server/server_policy.py \
    --ckpt_path $YOUR_CKPT --port $PORT --use_bf16 --cuda 0 \
    --lds_path $LDS > /tmp/vla_server_posanalysis.log 2>&1 &
SPID=$!
echo "Server PID: $SPID"

el=0
until grep -q "server listening" /tmp/vla_server_posanalysis.log 2>/dev/null; do
    sleep 2; el=$((el+2))
    if [ $el -ge 180 ]; then echo "server timeout"; cat /tmp/vla_server_posanalysis.log | tail; kill $SPID; exit 1; fi
done
grep "KF\|EKF" /tmp/vla_server_posanalysis.log | tail

# Run eval — only task 4, 50 episodes, log positions
$sim_python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path $YOUR_CKPT --args.host "127.0.0.1" --args.port $PORT \
    --args.task-suite-name libero_10 --args.num-trials-per-task 50 \
    --args.video-out-path "$OUT" --args.seed 7 --args.with_state "true" \
    --args.log-positions --args.only-task-id 4 \
    2>&1 | tee "$OUT/eval.log" | grep -E "Success|Current"

kill $SPID 2>/dev/null || true; sleep 3

# Analyze
echo ""
echo "════════ Final position analysis ════════"
python3 <<EOF
import json, numpy as np
path = "$OUT/final_positions.jsonl"
data = [json.loads(l) for l in open(path)]
print(f"Total episodes: {len(data)}  (success: {sum(d['done'] for d in data)}, fail: {sum(1-d['done'] for d in data)})")
print()
TOL = 0.03  # 3 cm
print(f"{'Ep':>3s} {'done':>5s}  {'porcelain (white→L)':>25s}  {'yellow_white (→R)':>22s}")
print(f"{'':>3s} {'':>5s}  {'xy_cm':>8s} {'z_cm':>5s} {'on?':>5s}  {'xy_cm':>8s} {'z_cm':>5s} {'on?':>5s}")
print("-"*78)
for d in data:
    p = d["positions"]
    if "porcelain_mug_1" not in p or "plate_1" not in p: continue
    m1, pl1 = np.array(p["porcelain_mug_1"]), np.array(p["plate_1"])
    m2, pl2 = np.array(p["white_yellow_mug_1"]), np.array(p["plate_2"])
    d1 = np.linalg.norm(m1[:2]-pl1[:2])*100; z1 = (m1[2]-pl1[2])*100
    d2 = np.linalg.norm(m2[:2]-pl2[:2])*100; z2 = (m2[2]-pl2[2])*100
    on1 = "✓" if d1 < 3 and z1 <= 0 else "✗"
    on2 = "✓" if d2 < 3 and z2 <= 0 else "✗"
    print(f"{d['episode']:>3d} {str(d['done']):>5s}  {d1:>7.1f}  {z1:>4.1f} {on1:>5s}  {d2:>7.1f}  {z2:>4.1f} {on2:>5s}")

# Summary for failures
fails = [d for d in data if not d["done"]]
if fails:
    print(f"\n=== Failure breakdown ({len(fails)} episodes) ===")
    near_miss_1 = 0; near_miss_2 = 0; way_off_1 = 0; way_off_2 = 0
    for d in fails:
        p = d["positions"]
        m1, pl1 = np.array(p["porcelain_mug_1"]), np.array(p["plate_1"])
        m2, pl2 = np.array(p["white_yellow_mug_1"]), np.array(p["plate_2"])
        d1 = np.linalg.norm(m1[:2]-pl1[:2])*100
        d2 = np.linalg.norm(m2[:2]-pl2[:2])*100
        if 3 <= d1 <= 6: near_miss_1 += 1
        if d1 > 6: way_off_1 += 1
        if 3 <= d2 <= 6: near_miss_2 += 1
        if d2 > 6: way_off_2 += 1
    print(f"  white_mug (LEFT plate):    near-miss 3-6cm: {near_miss_1},  way-off >6cm: {way_off_1}")
    print(f"  yellow_white (RIGHT plate): near-miss 3-6cm: {near_miss_2},  way-off >6cm: {way_off_2}")
EOF
