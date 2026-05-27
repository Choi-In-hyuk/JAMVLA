#!/bin/bash
# EMA eval script — VLA-JEPA + EMA token smoothing
# Usage: bash eval_libero_ema.sh [task_suite] [ema_alpha]

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
ema_alpha=${2:-0.5}
port=15086
num_trials_per_task=50
with_state="true"

folder_name="EMA_$(echo ${ema_alpha} | tr '.' 'p')"
video_out_path="results/${task_suite_name}/${folder_name}"
mkdir -p ${video_out_path}

echo "Task suite : ${task_suite_name}"
echo "EMA alpha  : ${ema_alpha}"
echo "Output     : ${video_out_path}"

fuser -k ${port}/tcp 2>/dev/null || true; sleep 1

rm -f /tmp/vla_server_ema.log
${sim_python} ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda 0 \
    --ema_alpha ${ema_alpha} > /tmp/vla_server_ema.log 2>&1 &
SERVER_PID=$!
echo "Policy server PID: ${SERVER_PID}"

echo "Waiting for server to be ready..."
elapsed=0
until grep -q "server listening" /tmp/vla_server_ema.log 2>/dev/null; do
    sleep 2; elapsed=$((elapsed + 2))
    if [ $elapsed -ge 120 ]; then
        echo "ERROR: Server failed to start. Log:"
        cat /tmp/vla_server_ema.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done
echo "Server is up (EMA enabled, alpha=${ema_alpha})."

${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "127.0.0.1" \
    --args.port ${port} \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task ${num_trials_per_task} \
    --args.video-out-path "${video_out_path}" \
    --args.with_state "${with_state}" \
    2>&1 | tee "${video_out_path}/eval.log"

kill ${SERVER_PID} 2>/dev/null
echo "Done. Results in ${video_out_path}/eval.log"
