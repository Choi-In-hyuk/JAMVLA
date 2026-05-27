#!/bin/bash


export LIBERO_HOME=/path/to/LIBERO-plus           # TODO: set to your LIBERO-Plus path
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/path/to/libero_env/bin/python  # TODO: set to your LIBERO conda env python

your_ckpt=/path/to/VLA-JEPA-LIBERO.pt             # TODO: set to your checkpoint path

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise")
task_suite_name=libero_mix

host="127.0.0.1"
base_port=14082
unnorm_key="franka"
index=0
with_state="true"

for perturbation_name in "${items[@]}"
do
perturbation_file_name=${perturbation_name// /_}
index=$((index+1))
port=$((base_port+index))

python ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda ${index} &

# TODO: perturbations should be modified in <LIBERO_HOME>/libero/libero/benchmark/__init__.py
num_trials_per_task=1 # must be 1 for perturbation evaluation
video_out_path="results/plus_${task_suite_name}/${perturbation_file_name}/${folder_name}"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
mkdir -p ${video_out_path}

# export DEBUG=true

${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port ${port} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" > "${video_out_path}/eval.log" \
    --args.category_value "$perturbation_name" \
    --args.with_state "$with_state" &
done
    