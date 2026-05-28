# 환경 세팅 가이드

새 머신에서 Mamba-JEPA-Flow를 학습하기 위한 단계별 가이드입니다.

---

## 1. 시스템 요구사항

- **GPU**: NVIDIA GPU (CUDA 12.4+ 권장). Blackwell 사용 시 CUDA 12.6+ 필요
- **GPU 메모리**: 최소 48GB (단일 GPU) 또는 다중 GPU
- **디스크**: 모델 + 데이터셋 합쳐서 약 50GB 여유 필요
- **OS**: Linux (Ubuntu 22.04 권장)

---

## 2. 코드 클론

```bash
git clone git@github.com:Choi-In-hyuk/JAMVLA.git
cd JAMVLA
```

---

## 3. Python 환경 + 의존성

### Conda 환경 생성
```bash
conda create -n vla_jepa python=3.10 -y
conda activate vla_jepa
```

### PyTorch 설치
```bash
# CUDA 12.4 (RTX A6000 등)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Blackwell (Pro 6000) - CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 프로젝트 의존성
```bash
pip install -r requirements.txt
```

### Mamba-SSM 컴파일 확인
```bash
python -c "from mamba_ssm import Mamba; print('Mamba OK')"
```

CUDA 컴파일 에러 시:
```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

---

## 4. 모델 가중치 다운로드

HuggingFace 모델을 로컬에 받습니다. 경로는 `~/data/models/`을 예시로 사용합니다.

```bash
mkdir -p ~/data/models && cd ~/data/models

# Qwen3-VL-2B-Instruct
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir Qwen3-VL-2B-Instruct

# V-JEPA 2 (ViT-L, 256px, 64 frames)
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 \
  --local-dir vjepa2-vitl-fpc64-256
```

---

## 5. 데이터셋 다운로드

### LIBERO (LeRobot 포맷)

4개 suite 모두 받습니다. 약 4-5GB.

```bash
mkdir -p ~/data/datasets/LIBERO && cd ~/data/datasets/LIBERO

for suite in libero_10 libero_goal libero_object libero_spatial; do
  huggingface-cli download IPEC-COMMUNITY/${suite}_no_noops_1.0.0_lerobot \
    --repo-type dataset \
    --local-dir ${suite}_no_noops_1.0.0_lerobot
done
```

### SSv2 (Something-Something V2)

약 19GB, 압축 풀면 더 큼.

```bash
mkdir -p ~/data/datasets/ssv2 && cd ~/data/datasets/ssv2

# 다운로드
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='morpheushoc/something-something-v2',
    repo_type='dataset',
    local_dir='.',
    resume_download=True,
)
"

# 비디오 압축 해제
cd videos
mkdir -p 20bn-something-something-v2
for f in 20bn-something-something-v2-*; do
  tar -xzf "$f" -C .
done
```

압축 해제 후 비디오는 `~/data/datasets/ssv2/videos/20bn-something-something-v2/`에 webm 파일들로 존재해야 합니다.

---

## 6. Config 경로 수정

새 머신의 경로에 맞게 두 config 파일을 수정합니다.

### `scripts/config/mamba_jepa_flow_stage1.yaml`
```yaml
framework:
  qwenvl:
    base_vlm: <새 경로>/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: <새 경로>/vjepa2-vitl-fpc64-256

datasets:
  video_data:
    video_dir: <새 경로>/ssv2/videos/20bn-something-something-v2
    text_file: <새 경로>/ssv2/test-answers.csv
```

### `scripts/config/mamba_jepa_flow.yaml`
```yaml
framework:
  qwenvl:
    base_vlm: <새 경로>/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: <새 경로>/vjepa2-vitl-fpc64-256

datasets:
  vla_data:
    data_root_dir: <새 경로>/datasets/LIBERO
  video_data:
    video_dir: <새 경로>/ssv2/videos/20bn-something-something-v2
    text_file: <새 경로>/ssv2/test-answers.csv
```

---

## 7. 다중 GPU 설정 (Blackwell Pro 6000 2장 사용 시)

`scripts/mamba_jepa_flow_full_train.sh`의 `--num_processes 1`을 `--num_processes 2`로 변경:

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \  # ← 2장 사용
  ./starVLA/training/train_vlajepa_video.py \
  --config_yaml ./scripts/config/mamba_jepa_flow_stage1.yaml
```

`scripts/config/mamba_jepa_flow_stage1.yaml`과 `scripts/config/mamba_jepa_flow.yaml`에서:
- `per_device_batch_size`를 늘리기 (예: 4 → 8 또는 16)
- `gradient_accumulation_steps`는 줄이기 (예: 4 → 1)
- `freeze_modules`에서 `qwen_vl_interface` 제거 (메모리 여유 있으니 학습 가능)

---

## 8. 동작 확인 (Forward Pass 테스트)

학습 전 모델이 잘 빌드되는지 확인:

```bash
cd JAMVLA
python -c "
from omegaconf import OmegaConf
from starVLA.model.framework.Mamba_JEPA_Flow import Mamba_JEPA_Flow
import torch, numpy as np
from PIL import Image

cfg = OmegaConf.load('./scripts/config/mamba_jepa_flow.yaml')
model = Mamba_JEPA_Flow(cfg).cuda()

image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
video = np.random.randint(0, 255, (2, 8, 224, 224, 3), dtype=np.uint8)
sample = {
    'action': np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
    'image': [image, image],
    'video': video,
    'lang': 'Pick up the red block.',
    'state': np.random.uniform(-1, 1, size=(1, 8)).astype(np.float32),
}
output = model([sample, sample])
print(f'action_loss={output[\"action_loss\"].item():.4f}, wm_loss={output[\"wm_loss\"].item():.4f}')
print('Forward pass OK')
"
```

---

## 9. 학습 실행

### 전체 파이프라인 (Stage 1 → Stage 2 자동)
```bash
mkdir -p logs
nohup bash scripts/mamba_jepa_flow_full_train.sh > logs/train.log 2>&1 &
echo "PID: $!"
```

### 로그 확인
```bash
# 실시간
tail -f logs/train.log

# Loss만
grep "Step.*Loss" logs/train.log | tail -20

# 에러
grep -i "error\|FAILED\|OOM" logs/train.log
```

---

## 10. 평가 (LIBERO-10)

### LIBERO 시뮬레이터 설치
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### 평가 실행
```bash
python examples/LIBERO/eval_libero.py \
  --pretrained_path checkpoints/mamba_jepa_flow/checkpoints/steps_50000_pytorch_model.pt \
  --task_suite_name libero_10
```

---

## 문제 해결

### NCCL 에러 (네트워크 인터페이스)
`scripts/mamba_jepa_flow_full_train.sh`의 `NCCL_SOCKET_IFNAME=lo`를 시스템 인터페이스에 맞게 변경:
```bash
ip link show | grep -E "^[0-9]" | awk '{print $2}' | tr -d ':'
```

### CUDA OOM
- `per_device_batch_size`를 줄임
- `gradient_accumulation_steps`를 늘림
- `freeze_modules`에 `qwen_vl_interface` 추가
- `enable_gradient_checkpointing: true` 확인

### 포트 충돌 (29500)
이전 학습 프로세스가 살아있는 경우:
```bash
pkill -9 -f "python.*train\|accelerate\|deepspeed"
```

### GPU 메모리 좀비
프로세스는 사라졌지만 GPU 메모리가 해제되지 않을 때:
```bash
sudo nvidia-smi --gpu-reset
# 실패 시 재부팅 필요
```
