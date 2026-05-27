#!/bin/bash
# Single-GPU eval script (A6000)
# Usage: bash eval_libero_single_gpu.sh [task_suite]
#   task_suite: libero_10 | libero_goal | libero_object | libero_spatial (default: libero_spatial)

export PYTHONDONTWRITEBYTECODE=1

# Fix: vla_jepa env torch needs nvidia libs from vjepa2 env
_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/home/choi/miniconda3/envs/vla_jepa/bin/python

your_ckpt=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
task_suite_name=${1:-libero_spatial}
port=15083
num_trials_per_task=50
with_state="true"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
video_out_path="results/${task_suite_name}/${folder_name}"
LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
mkdir -p ${video_out_path}

echo "Task suite : ${task_suite_name}"
echo "Checkpoint : ${your_ckpt}"
echo "Output     : ${video_out_path}"

# Kill any leftover server on this port
fuser -k ${port}/tcp 2>/dev/null; sleep 1

# Start policy server on GPU 0
rm -f /tmp/vla_server.log
python ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda 0 > /tmp/vla_server.log 2>&1 &
SERVER_PID=$!
echo "Policy server PID: ${SERVER_PID}"

echo "Waiting for server to be ready..."
until grep -q "server listening" /tmp/vla_server.log 2>/dev/null; do sleep 2; done
echo "Server is up."

# Run evaluation
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
