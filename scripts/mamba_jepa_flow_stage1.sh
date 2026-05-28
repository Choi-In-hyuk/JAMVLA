export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  ./starVLA/training/train_vlajepa_video.py \
  --config_yaml ./scripts/config/mamba_jepa_flow_stage1.yaml
