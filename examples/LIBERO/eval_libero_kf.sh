#!/bin/bash
# KF eval script (A6000) — VLA-JEPA + Learned LDS KF on embodied_action_tokens
# Usage: bash eval_libero_kf.sh [task_suite] [lds_path]

export PYTHONDONTWRITEBYTECODE=1

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

your_ckpt=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
task_suite_name=${1:-libero_spatial}
lds_path=${2:-/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_spatial_lds_tokens/lds_64.npz}
port=15084
num_trials_per_task=50
with_state="true"

folder_name="KF_$(basename ${lds_path} .npz)"
video_out_path="results/${task_suite_name}/${folder_name}"
mkdir -p ${video_out_path}

echo "Task suite : ${task_suite_name}"
echo "Checkpoint : ${your_ckpt}"
echo "LDS path   : ${lds_path}"
echo "Output     : ${video_out_path}"

if [ ! -f "${lds_path}" ]; then
    echo "ERROR: LDS checkpoint not found: ${lds_path}"
    exit 1
fi

fuser -k ${port}/tcp 2>/dev/null; sleep 1

rm -f /tmp/vla_server_kf.log
python ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda 0 \
    --lds_path ${lds_path} \
    --kf_q 0.1 \
    --kf_r 5.0 > /tmp/vla_server_kf.log 2>&1 &
SERVER_PID=$!
echo "Policy server PID: ${SERVER_PID}"

echo "Waiting for server to be ready..."
elapsed=0
until grep -q "server listening" /tmp/vla_server_kf.log 2>/dev/null; do
    sleep 2; elapsed=$((elapsed + 2))
    if [ $elapsed -ge 120 ]; then
        echo "ERROR: Server failed to start within 120s. Log:"
        cat /tmp/vla_server_kf.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done
echo "Server is up (KF enabled)."

${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "127.0.0.1" \
    --args.port ${port} \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task ${num_trials_per_task} \
    --args.video-out-path "${video_out_path}" \
    --args.with_state "${with_state}" \
    2>&1 | tee "${video_out_path}/eval.log"

kill ${SERVER_PID}
echo "Done. Results in ${video_out_path}/eval.log"
