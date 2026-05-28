# Mamba-JEPA-Flow

**세계 모델의 예측 오차 보정과 상태 공간 시간 추론을 결합한 예측형 제어 VLA**

미래를 예측하고 (V-JEPA), 예측 오차를 실시간으로 감지하며 (Mamba), 부드러운 보정 행동을 생성하는 (Flow-Matching) 로봇 조작 모델입니다.

---

## 연구 동기

현재 VLA 모델들은 근본적으로 **반응형**입니다. 관측을 행동으로 매핑할 뿐, 자신의 행동이 물리 세계에 미칠 영향을 이해하지 못합니다. 로봇 팔이 방해받거나 물체가 미끄러지면, 모델은 *예측*을 학습한 적이 없기 때문에 실패합니다.

Mamba-JEPA-Flow는 세 가지 핵심 한계를 해결합니다:

| 문제 | 기존 접근법 | 본 연구의 해결 방식 |
|------|-----------|-------------------|
| 물리 법칙 이해 부재 | 모방 학습 (보고 → 행동) | V-JEPA 세계 모델이 잠재 공간에서 미래 상태를 예측 |
| 오차 보정 능력 부재 | 개방 루프(Open-loop) 행동 실행 | Mamba가 예측 오차 (ŝ\_t vs s\_t)를 감지하고 재계획 |
| 느린 추론 / 짧은 기억 | Transformer O(N²) 어텐션 | Mamba SSM이 O(N) 선형 시간으로 시간적 맥락 처리 |

핵심 통찰: 로봇은 인간의 소뇌처럼 **예측 → 비교 → 수정**해야 하며, 단순히 반응해서는 안 됩니다.

---

## 아키텍처

```
                        ┌─────────────────────────────────────────────┐
                        │          Phase 0: Qwen-VL (동결)            │
 언어 명령 + 이미지 ───▶│  "빨간 블록을 집어"  +  카메라 영상          │
                        │           ↓                                │
                        │     action_tokens (잠재 행동 의도)          │
                        └──────────┬──────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        │
┌──────────────────┐   ┌────────────────────┐              │
│  Phase 1: V-JEPA │   │  JEPA Predictor    │              │
│ Encoder (동결)   │   │  (학습 가능)       │              │
│                  │   │                    │              │
│ 프레임 → s_{t-1} │   │ s_{t-1} + a_{t-1} │              │
│           s_t    │   │      → ŝ_t        │              │
└────────┬─────────┘   └────────┬───────────┘              │
         │                      │                          │
         ▼                      ▼ (.detach())              ▼
┌──────────────────────────────────────────────────────────────────┐
│              Phase 2: Mamba 시공간 인터리버 (핵심 기여)           │
│                                                                  │
│  입력: [ s_{t-1},  a_{t-1},  ŝ_t,  s_t ] (시간순 교차 배열)    │
│           ↓         ↓         ↓      ↓                          │
│         [h_1] ──▶ [h_2] ──▶ [h_3] ──▶ [h_4] ──▶ [readout]    │
│                              ▲                                   │
│                        예측 오차 인지                             │
│                       (ŝ_t vs s_t)                              │
│                                                                  │
│  출력: 컨텍스트 벡터 c (32개 토큰)                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│          Phase 3: Flow-Matching 액션 디코더                      │
│                                                                  │
│  조건: c (무엇을 할지) + p_t (로봇 관절 상태, Late Injection)    │
│                                                                  │
│  가우시안 노이즈 ──▶ ODE: dz/dt = v_θ(z,t,c,p_t) ──▶ a_t      │
│                     (연속적이고 부드러운 행동 궤적)               │
└──────────────────────────────────────────────────────────────────┘
```

### 설계 원칙

- **Gradient 분리**: `predicted_states.detach()`로 `action_loss`가 세계 모델을 오염시키지 않도록 차단. Predictor는 오직 `wm_loss`로만 학습.
- **Late Injection**: 로봇 고유 상태(`p_t`)는 Flow-Matching 디코더에서만 주입. Mamba 백본은 하드웨어에 독립적이므로, 디코더만 교체하면 다른 로봇에 이식 가능.
- **언어 정보 중복 제거**: 언어는 Qwen-VL에서 한 번만 처리. Mamba는 순수하게 V-JEPA 잠재 공간에서만 동작.

---

## 학습

### Co-training 전략 (VLA-JEPA 원본 방식)

매 학습 반복(iteration)마다 두 단계를 번갈아 수행합니다:

**Step 1 — 로봇 데이터 (LIBERO)**
```
action_loss (Flow-Matching 속도장 예측) + 0.1 * wm_loss (JEPA 상태 예측)
→ 업데이트 대상: Mamba, Action Model, VJ Predictor
```

**Step 2 — 비디오 데이터 (Something-Something V2)**
```
wm_loss만 (행동 레이블 없음)
→ 업데이트 대상: VJ Predictor만 (물리 이해력 보충)
```

### 동결 vs 학습 모듈

| 모듈 | 파라미터 수 | 상태 | 업데이트 기준 |
|------|-----------|------|-------------|
| Qwen-VL | ~3B | 동결 | — |
| V-JEPA Encoder | ~300M | 동결 | — |
| VJ Predictor | 162M | 학습 | wm_loss |
| Mamba Backbone | 35M | 학습 | action_loss |
| Flow-Matching Head | 155M | 학습 | action_loss |

### 학습 실행

```bash
# 단일 GPU (A6000 48GB)
bash scripts/mamba_jepa_flow_train.sh
```

### 주요 설정

모든 하이퍼파라미터는 `scripts/config/mamba_jepa_flow.yaml`에 정의되어 있습니다.

```yaml
framework:
  mamba_backbone:
    d_model: 1024          # Mamba 내부 차원
    n_layers: 4            # Mamba 레이어 수
    num_output_tokens: 32  # Flow-Matching에 전달할 readout 토큰 수
    output_dim: 2048       # DiT cross_attention_dim과 일치

trainer:
  learning_rate:
    mamba_backbone: 1.0e-04
    action_model: 1.0e-04
  freeze_modules: 'vj_encoder,qwen_vl_interface'
```

---

## 평가

LIBERO-10 벤치마크 (10개 장기 조작 태스크)에서 평가:

```bash
python examples/LIBERO/eval_libero.py \
  --pretrained_path checkpoints/mamba_jepa_flow/checkpoints/<step>_pytorch_model.pt \
  --task_suite_name libero_10
```

---

## 프로젝트 구조

```
starVLA/
├── model/
│   ├── framework/
│   │   ├── Mamba_JEPA_Flow.py          # 메인 프레임워크 (Phase 0-3)
│   │   └── VLA_JEPA.py                 # 원본 VLA-JEPA (참조용)
│   └── modules/
│       ├── mamba_backbone/
│       │   └── mamba_temporal_interleaver.py  # Mamba SSM 인터리버
│       ├── action_model/
│       │   └── GR00T_ActionHeader.py         # Flow-Matching 디코더
│       └── world_model/
│           └── vj2_predictor.py              # V-JEPA Predictor
scripts/
├── config/
│   └── mamba_jepa_flow.yaml            # 학습 설정
└── mamba_jepa_flow_train.sh            # 실행 스크립트
```

---

## 핵심 참고 논문

- **VLA-JEPA**: Joint Embedding Predictive Architecture 기반 로봇 제어용 잠재 세계 모델
- **Mamba**: O(N) 선형 시간 선택적 상태 공간 모델
- **Flow-Matching**: 최적 수송 ODE 기반 연속 행동 생성
- **GR00T N1.5**: Flow-matching 액션 헤드 아키텍처
