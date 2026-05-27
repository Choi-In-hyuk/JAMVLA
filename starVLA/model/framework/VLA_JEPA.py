# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # KF state (populated by load_kf; None = KF disabled)
        self._lds          = None
        self._kf_mode      = None   # "KF" | "EKF" | "DKF"
        self._kf_nonlinear = False
        self._kf_z         = None   # (latent_dim,) numpy
        self._kf_P         = None   # (latent_dim, latent_dim) numpy
        self._kf_Q         = None
        self._kf_R         = None
        # Placement-aware adaptive KF (set via load_kf)
        self._adaptive          = False
        self._placement_alpha   = 1.0   # 1.0 = full KF, 0.0 = pass-through
        self._opening_threshold = 0.001 # gripper qpos delta to trigger
        self._last_gripper      = None  # scalar (mean of 2 fingers)
        # Adaptive observation noise R_t from token spread (set via enable_adaptive_r)
        self._adaptive_r        = False
        self._ar_gamma          = 1.0   # sensitivity of R to spread ratio
        self._ar_clip           = (0.5, 2.0)
        self._ar_spread_ema     = None  # running mean of token spread
        self._ar_beta           = 0.9   # EMA decay for spread normalization
        # Approach-aware KF (weakens KF during precision phases — set via enable_approach_aware)
        self._approach_aware    = False
        self._app_R_scale       = 5.0   # multiply R during precision phase (smoothing↓)
        self._app_slow_speed    = 0.01  # ||Δpos|| threshold per step (low = precision)
        self._app_gripper_close = 0.03  # gripper qpos below this = holding object
        self._last_ee_pos       = None

        # EMA state (populated by load_ema; None = EMA disabled)
        self._ema_alpha = None
        self._ema_y     = None   # (feat_dim,) numpy per batch item

        # Per-task LDS bank (populated by load_kf_per_task)
        self._per_task      = False
        self._lds_bank      = {}    # normalized_task_name → LearnedLDS
        self._active_task   = None  # currently loaded task key

    def load_kf(self, lds_path: str, q_noise: float | None = None, r_noise: float | None = None) -> None:
        """Load a trained KF model. Auto-detects:
          - DKF       (filename contains 'dkf_' OR sibling '*_nets.pt' present)
          - Nonlinear (sibling '*_mlp.pt' present)
          - Linear    (default)
        """
        import sys, os
        sys.path.insert(0, "/home/choi/vjepa2")

        dkf_nets_path = lds_path.replace(".npz", "_nets.pt")
        mlp_path      = lds_path.replace(".npz", "_mlp.pt")

        if os.path.exists(dkf_nets_path):
            from src.models.dkf.token_dkf import TokenDKF
            self._lds = TokenDKF.load(lds_path)
            self._kf_mode = "DKF"
            self._kf_nonlinear = False
        elif os.path.exists(mlp_path):
            from src.models.kf.nonlinear_lds import NonlinearLDS
            self._lds = NonlinearLDS.load(lds_path)
            self._kf_mode = "EKF"
            self._kf_nonlinear = True
        else:
            from src.models.kf.learned_lds import LearnedLDS
            self._lds = LearnedLDS.load(lds_path)
            self._kf_mode = "KF"
            self._kf_nonlinear = False

        d = self._lds.latent_dim

        # Resolve Q, R: explicit > EM-learned > manual default
        if q_noise is None:
            q_noise = self._lds.q_em if self._lds.q_em is not None else 0.1
            q_src = "EM" if self._lds.q_em is not None else "default"
        else:
            q_src = "explicit"
        if r_noise is None:
            r_noise = self._lds.r_em if self._lds.r_em is not None else 5.0
            r_src = "EM" if self._lds.r_em is not None else "default"
        else:
            r_src = "explicit"

        self._kf_Q = q_noise * np.eye(d)
        self._kf_R = r_noise * np.eye(d)
        self.reset_kf()

        logger.info(
            f"[{self._kf_mode}] Loaded model from {lds_path}  latent_dim={d}  "
            f"q={q_noise:.4f}({q_src})  r={r_noise:.4f}({r_src})"
        )

    @staticmethod
    def _normalize_task(s: str) -> str:
        """Normalize a task string for matching (lowercase, alnum+underscore only)."""
        import re
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def load_kf_per_task(self, ckpt_dir: str) -> None:
        """Load a bank of per-task LDS (lds_em_task{NN}.npz + task_index.json).

        At inference, the LDS is auto-selected by matching the instruction string
        against the task names in task_index.json. Falls back to no-op if no match.
        """
        import sys, json, os
        sys.path.insert(0, "/home/choi/vjepa2")
        from src.models.kf.learned_lds import LearnedLDS

        with open(os.path.join(ckpt_dir, "task_index.json")) as f:
            index = json.load(f)
        self._lds_bank = {}
        for task_name, idx in index.items():
            path = os.path.join(ckpt_dir, f"lds_em_task{idx:02d}.npz")
            if os.path.exists(path):
                self._lds_bank[self._normalize_task(task_name)] = LearnedLDS.load(path)
        self._per_task   = True
        self._kf_mode    = "KF"
        self._kf_nonlinear = False
        self._active_task = None
        logger.info(f"[Per-task KF] Loaded {len(self._lds_bank)} task-specific LDS from {ckpt_dir}")

    def _select_task_lds(self, instruction: str) -> None:
        """Switch active LDS to the one matching the instruction (substring match)."""
        if not self._per_task:
            return
        norm = self._normalize_task(instruction)
        # Find a bank key that is contained in (or contains) the instruction
        best = None
        for key in self._lds_bank:
            if norm in key or key in norm or self._token_overlap(norm, key) >= 0.6:
                best = key
                break
        if best is not None and best != self._active_task:
            self._lds = self._lds_bank[best]
            d = self._lds.latent_dim
            self._kf_Q = (self._lds.q_em if self._lds.q_em else 0.1) * np.eye(d)
            self._kf_R = (self._lds.r_em if self._lds.r_em else 5.0) * np.eye(d)
            self._active_task = best
            self.reset_kf()
            logger.info(f"[Per-task KF] switched to '{best}' (q={self._kf_Q[0,0]:.2f} r={self._kf_R[0,0]:.2f})")

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        """Word-level Jaccard overlap of two normalized strings."""
        sa, sb = set(a.split("_")), set(b.split("_"))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def enable_approach_aware(self, R_scale: float = 5.0,
                              slow_speed: float = 0.01,
                              gripper_close: float = 0.03) -> None:
        """Approach-aware KF: weaken smoothing during precision phases
        (slow EE motion while holding an object) — addresses the regression mode
        where KF lag during slow approach drifts placement by a few mm.

        R_t = R * R_scale  when (||Δpos|| < slow_speed AND gripper_qpos < gripper_close)
        R_t = R            otherwise
        """
        self._approach_aware    = True
        self._app_R_scale       = float(R_scale)
        self._app_slow_speed    = float(slow_speed)
        self._app_gripper_close = float(gripper_close)
        self._last_ee_pos       = None
        logger.info(f"[Approach-aware] R_scale={R_scale} slow_speed={slow_speed} "
                    f"gripper_close={gripper_close}")

    def enable_adaptive_r(self, gamma: float = 1.0,
                          clip_lo: float = 0.5, clip_hi: float = 2.0,
                          beta: float = 0.9) -> None:
        """Enable per-step adaptive observation noise R_t from token spread.

        At each step, R_t = r_em * clip( (spread_t / EMA(spread))**gamma, lo, hi ).
        High token spread (the 32 embodied tokens disagree → low confidence) raises R
        → more smoothing; low spread (confident) lowers R → trust the observation.
        """
        self._adaptive_r    = True
        self._ar_gamma      = float(gamma)
        self._ar_clip       = (float(clip_lo), float(clip_hi))
        self._ar_beta       = float(beta)
        self._ar_spread_ema = None
        logger.info(f"[Adaptive-R] gamma={gamma} clip=({clip_lo},{clip_hi}) beta={beta}")

    def enable_adaptive_placement(self, placement_alpha: float = 0.2,
                                   opening_threshold: float = 0.001) -> None:
        """Enable placement-aware KF: reduce smoothing when gripper opens.

        At release moments, blend output as
            y = alpha * y_filtered + (1 - alpha) * y_obs
        where alpha = placement_alpha (< 1.0) only when gripper is opening.
        """
        self._adaptive          = True
        self._placement_alpha   = float(placement_alpha)
        self._opening_threshold = float(opening_threshold)
        self._last_gripper      = None
        logger.info(f"[Adaptive] placement_alpha={placement_alpha} "
                    f"opening_threshold={opening_threshold}")

    def reset_kf(self) -> None:
        """Reset KF state to uninformed prior (call at episode start)."""
        if self._lds is None:
            return
        d          = self._lds.latent_dim
        self._kf_z = np.zeros(d, dtype=np.float32)
        self._kf_P = np.eye(d, dtype=np.float32)
        self._last_gripper = None
        self._last_ee_pos  = None

    def load_ema(self, alpha: float) -> None:
        """Enable EMA smoothing on embodied_action_tokens (no offline training needed)."""
        self._ema_alpha = alpha
        self._ema_y     = None
        logger.info(f"[EMA] Enabled  alpha={alpha}")

    def reset_ema(self) -> None:
        """Reset EMA state (call at episode start)."""
        self._ema_y = None

    def _ema_step(self, y_obs: np.ndarray) -> np.ndarray:
        """One EMA step.  y_obs: (feat_dim,) → returns smoothed (feat_dim,)."""
        if self._ema_y is None:
            self._ema_y = y_obs.copy()
        else:
            self._ema_y = self._ema_alpha * y_obs + (1 - self._ema_alpha) * self._ema_y
        return self._ema_y.copy()

    def _kf_step(self, y_obs: np.ndarray, state: np.ndarray | None = None,
                 spread: float | None = None) -> np.ndarray:
        """One filter step.  y_obs: (feat_dim,) → filtered (feat_dim,).
        If state (8-dim) + placement-adaptive on, reduce KF strength on gripper-open.
        If spread (token disagreement) + adaptive-R on, modulate R_t per step.
        """
        z_obs = y_obs @ self._lds.E.T                                # encode to latent

        # Per-step adaptive observation noise from token spread
        R_eff = self._kf_R
        # Approach-aware: weaken KF during slow precision motion while holding object
        if self._approach_aware and state is not None and len(state) >= 8:
            ee_pos = np.asarray(state[:3], dtype=np.float32)
            gripper = float((state[6] + state[7]) / 2.0)
            speed = float(np.linalg.norm(ee_pos - self._last_ee_pos)) \
                    if self._last_ee_pos is not None else 1.0
            self._last_ee_pos = ee_pos.copy()
            if speed < self._app_slow_speed and gripper < self._app_gripper_close:
                R_eff = self._kf_R * self._app_R_scale  # precision phase → weaker KF
        if self._adaptive_r and spread is not None:
            if self._ar_spread_ema is None:
                self._ar_spread_ema = spread
            ratio = spread / (self._ar_spread_ema + 1e-8)
            scale = float(np.clip(ratio ** self._ar_gamma, *self._ar_clip))
            R_eff = self._kf_R * scale
            self._ar_spread_ema = self._ar_beta * self._ar_spread_ema + (1 - self._ar_beta) * spread

        # DKF: neural prior/inference fusion (no explicit covariance)
        if self._kf_mode == "DKF":
            self._kf_z = self._lds.kf_step(z_obs, self._kf_z)
            y_filtered = self._kf_z @ self._lds.E
        else:
            # Predict (linear vs nonlinear)
            if self._kf_nonlinear:
                z_pred = self._lds.predict_mean(self._kf_z)          # A z + g(z)
                J      = self._lds.jacobian(self._kf_z)              # A + dg/dz
                P_pred = J @ self._kf_P @ J.T + self._kf_Q
            else:
                z_pred = self._lds.A @ self._kf_z
                P_pred = self._lds.A @ self._kf_P @ self._lds.A.T + self._kf_Q
            S = P_pred + R_eff
            K = P_pred @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
            self._kf_z = z_pred + K @ (z_obs - z_pred)
            self._kf_P = (np.eye(len(self._kf_z)) - K) @ P_pred
            y_filtered = self._kf_z @ self._lds.E

        # ── Placement-aware blending ──
        if self._adaptive and state is not None and len(state) >= 8:
            cur_g = float((state[6] + state[7]) / 2.0)
            if self._last_gripper is not None:
                opening = cur_g - self._last_gripper
                if opening > self._opening_threshold:
                    a = self._placement_alpha
                    self._last_gripper = cur_g
                    return a * y_filtered + (1.0 - a) * y_obs
            self._last_gripper = cur_g

        return y_filtered

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                input_videos.append(self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            #print(input_states.shape, action_tokens.shape)
            #exit()
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )
        
        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                #print(state.shape)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        # Per-task: select the task-specific LDS from the instruction
        if self._per_task and len(instructions) > 0:
            self._select_task_lds(instructions[0])

        # KF filtering on embodied_action_tokens (if LDS loaded)
        if self._lds is not None:
            if kwargs.get("reset_kf", False):
                self.reset_kf()
            tokens_np  = embodied_action_tokens.float().cpu().numpy()  # (B, n_tok, H)
            y_raw      = tokens_np.mean(axis=1)                        # (B, H) mean-pool
            # Per-step token spread (disagreement among the n_tok embodied tokens)
            spread_np  = tokens_np.var(axis=1).mean(axis=1)            # (B,) mean variance over H
            # Pass per-batch state to _kf_step for placement-aware blending
            state_np = np.array(state) if state is not None else None  # (B, 8) or None
            y_filtered = np.stack([
                self._kf_step(
                    y_raw[b],
                    state=(state_np[b] if state_np is not None else None),
                    spread=float(spread_np[b]),
                )
                for b in range(tokens_np.shape[0])
            ], axis=0)
            correction = torch.from_numpy(y_filtered - y_raw).to(embodied_action_tokens)
            embodied_action_tokens = embodied_action_tokens + correction.unsqueeze(1)

        # EMA smoothing on embodied_action_tokens (if EMA enabled)
        elif self._ema_alpha is not None:
            if kwargs.get("reset_kf", False):
                self.reset_ema()
            tokens_np  = embodied_action_tokens.float().cpu().numpy()  # (B, n_tok, H)
            y_raw      = tokens_np.mean(axis=1)                        # (B, H) mean-pool
            y_filtered = np.stack([self._ema_step(y_raw[b]) for b in range(tokens_np.shape[0])], axis=0)
            correction = torch.from_numpy(y_filtered - y_raw).to(embodied_action_tokens)
            embodied_action_tokens = embodied_action_tokens + correction.unsqueeze(1)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
