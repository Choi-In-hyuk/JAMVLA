from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.mamba_backbone.mamba_temporal_interleaver import MambaTemporalInterleaver
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Mamba_JEPA_Flow")
class Mamba_JEPA_Flow(baseframework):
    """
    Predictive control VLA combining V-JEPA world model, Mamba SSM temporal
    interleaver, and Flow-Matching action decoder.

    Data flow:
        Qwen-VL → action_tokens → JEPA Predictor → ŝ_t ─┐
                                                          ├→ Mamba → c → Flow-Matching(c, p_t) → a_t
                              V-JEPA Encoder → s_{t-1}, s_t ┘
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # ── Qwen-VL (language + vision fusion) ──
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        action_tokens, self.action_token_ids = self._expand_action_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
        )

        # ── V-JEPA Encoder + Predictor (world model) ──
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.tubelet_size = tubelet_size
        self.num_temporal_slots = self.config.framework.vj2_model.num_frames // tubelet_size

        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.num_temporal_slots,
            img_size=(self.vj_encoder.config.image_size, self.vj_encoder.config.image_size),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2,
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )

        # ── Mamba Temporal Interleaver (NEW) ──
        mamba_cfg = self.config.framework.mamba_backbone
        self.mamba_backbone = MambaTemporalInterleaver(
            state_dim=self.vj_encoder.config.hidden_size * 2,
            action_dim=self.qwen_vl_interface.model.config.hidden_size,
            d_model=mamba_cfg.d_model,
            n_layers=mamba_cfg.n_layers,
            d_state=mamba_cfg.d_state,
            d_conv=mamba_cfg.d_conv,
            expand=mamba_cfg.expand,
            num_output_tokens=mamba_cfg.num_output_tokens,
            output_dim=mamba_cfg.output_dim,
        )

        # ── Flow-Matching Action Head ──
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = mamba_cfg.output_dim
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size

        # ── Prompt templates ──
        num_act_tokens_per_step = self.config.framework.vj2_model.num_action_tokens_per_timestep
        self.replace_prompt = "".join(
            [tok * num_act_tokens_per_step for tok in action_tokens[:self.num_temporal_slots - 1]]
        )

    def _expand_action_tokenizer(self, tokenizer, special_action_token, max_action_tokens):
        """Add action tokens to the tokenizer. No embodied_action_token needed."""
        action_tokens, action_token_ids = [], []
        for i in range(max_action_tokens):
            tok = special_action_token.format(i)
            action_tokens.append(tok)
            if tok not in tokenizer.get_vocab():
                tokenizer.add_tokens([tok], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(tok))

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ; tokenizer vocab: {len(tokenizer)}")
        return action_tokens, action_token_ids

    def _encode_videos(self, batch_videos: np.ndarray) -> torch.Tensor:
        """Encode multi-view videos through V-JEPA encoder.

        Args:
            batch_videos: (B, V, T, 3, H, W) numpy array

        Returns:
            video_embeddings: (B, T_slots * patches_per_frame, embed_dim * 2)
        """
        B, V, T, C, H, W = batch_videos.shape
        flat_videos = batch_videos.reshape(B * V, T, C, H, W)
        input_videos = []
        for i in range(B * V):
            input_videos.append(
                self.vj_processor(videos=flat_videos[i], return_tensors="pt")
                ["pixel_values_videos"].to(self.vj_encoder.device)
            )
        input_videos = torch.cat(input_videos, dim=0)
        with torch.no_grad():
            video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
            video_embeddings = torch.cat(
                torch.chunk(video_embeddings, chunks=V, dim=0), dim=2
            )
        return video_embeddings

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [ex["image"] for ex in examples]
        batch_videos = [ex["video"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples] if "action" in examples[0] else None
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # ── Phase 0: Qwen-VL → action_tokens ──
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            )
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
            )

        action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
        )
        action_indices = action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)

        # ── Phase 1: V-JEPA Encoder + Predictor ──
        batch_videos_np = np.stack(batch_videos)
        batch_videos_np = batch_videos_np.transpose(0, 1, 2, 5, 3, 4)  # (B, V, T, 3, H, W)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_embeddings = self._encode_videos(batch_videos_np)
            # video_embeddings: (B, T_slots * patches_per_frame, embed_dim*2)

            T = self.num_temporal_slots
            patches_per_frame = video_embeddings.shape[1] // T

            input_states = video_embeddings[:, :patches_per_frame * (T - 1), :]
            gt_states = video_embeddings[:, patches_per_frame:, :]

            predicted_states = self.vj_predictor(input_states, action_tokens)

            wm_loss = F.l1_loss(predicted_states, gt_states, reduction="mean")

            # ── Phase 2: Mamba Temporal Interleaver ──
            all_states = video_embeddings.view(B, T, patches_per_frame, -1)
            pred_states_reshaped = predicted_states.view(B, T - 1, patches_per_frame, -1)

            num_act_tokens_per_step = self.config.framework.vj2_model.num_action_tokens_per_timestep
            act_tok_reshaped = action_tokens.view(B, T - 1, num_act_tokens_per_step, H)

            # detach: action_loss gradient does not flow back to vj_predictor
            # predictor is updated only by wm_loss (same as original VLA-JEPA)
            context = self.mamba_backbone(
                observed_states=all_states,
                action_tokens=act_tok_reshaped,
                predicted_states=pred_states_reshaped.detach(),
            )

        if actions is None:
            return {"wm_loss": wm_loss}

        # ── Phase 3: Flow-Matching Action Head ──
        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=context.device, dtype=context.dtype,
            )
            actions_target = actions_tensor[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer else 4
            )
            actions_target_rep = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            context_rep = context.repeat(repeated_diffusion_steps, 1, 1)

            state_rep = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=context.device, dtype=context.dtype,
                )
                state_rep = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(context_rep, actions_target_rep, state_rep)

        return {"action_loss": action_loss, "wm_loss": wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        batch_videos: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> dict:
        """
        Inference: requires batch_videos for V-JEPA encoding.

        Args:
            batch_images: List of [view1, view2] PIL images per sample.
            instructions: List of language instructions.
            batch_videos: (B, V, T, H, W, 3) numpy array. Required.
            state: (B, 1, state_dim) numpy array or None.
        """
        if batch_videos is None:
            raise ValueError(
                "Mamba_JEPA_Flow requires batch_videos at inference for V-JEPA encoding."
            )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # ── Phase 0: Qwen-VL → action_tokens ──
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt},
        )

        action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
        )
        action_indices = action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)

        # ── Phase 1: V-JEPA Encoder + Predictor ──
        batch_videos_np = batch_videos.transpose(0, 1, 2, 5, 3, 4) if batch_videos.shape[-1] == 3 else batch_videos

        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_embeddings = self._encode_videos(batch_videos_np)

            T = self.num_temporal_slots
            patches_per_frame = video_embeddings.shape[1] // T

            input_states = video_embeddings[:, :patches_per_frame * (T - 1), :]
            predicted_states = self.vj_predictor(input_states, action_tokens)

            # ── Phase 2: Mamba → context ──
            all_states = video_embeddings.view(B, T, patches_per_frame, -1)
            pred_states_reshaped = predicted_states.view(B, T - 1, patches_per_frame, -1)

            num_act_tokens_per_step = self.config.framework.vj2_model.num_action_tokens_per_timestep
            act_tok_reshaped = action_tokens.view(B, T - 1, num_act_tokens_per_step, H)

            context = self.mamba_backbone(
                observed_states=all_states,
                action_tokens=act_tok_reshaped,
                predicted_states=pred_states_reshaped,
            )

        # ── Phase 3: Flow-Matching → actions ──
        state_tensor = (
            torch.from_numpy(np.array(state)).to(context.device, dtype=context.dtype)
            if state is not None else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(context, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("./scripts/config/mamba_jepa_flow.yaml")
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = Mamba_JEPA_Flow(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    video = np.random.randint(0, 255, (2, 8, 224, 224, 3), dtype=np.uint8)
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "video": video,
        "lang": "Pick up the red block and place it on the blue block.",
        "state": np.random.uniform(-1, 1, size=(1, 8)).astype(np.float16),
    }
    batch = [sample, sample]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")
    print(f"WM Loss: {forward_output['wm_loss'].item()}")
