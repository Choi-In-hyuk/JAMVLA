import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaTemporalInterleaver(nn.Module):
    """
    Processes interleaved V-JEPA states, predicted states, and action tokens
    through Mamba SSM layers to produce a context vector for Flow-Matching.

    Input sequence (per timestep transition):
        [s_{t-1}, a_{t-1}, ŝ_t, s_t, a_t, ŝ_{t+1}, s_{t+1}, ...]
    Learnable readout tokens are appended at the end and extracted after
    Mamba processing to form the output context.
    """

    TOKEN_TYPE_OBSERVED = 0
    TOKEN_TYPE_ACTION = 1
    TOKEN_TYPE_PREDICTED = 2
    TOKEN_TYPE_READOUT = 3

    def __init__(
        self,
        state_dim: int = 2048,
        action_dim: int = 2048,
        d_model: int = 1024,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_output_tokens: int = 32,
        output_dim: int = 2048,
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pred_state_proj = nn.Linear(state_dim, d_model)

        self.token_type_embed = nn.Embedding(4, d_model)

        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.readout_tokens = nn.Parameter(torch.randn(num_output_tokens, d_model) * 0.02)

        self.output_proj = nn.Linear(d_model, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        observed_states: torch.Tensor,
        action_tokens: torch.Tensor,
        predicted_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            observed_states:  (B, T, patches, state_dim)  — V-JEPA encoder outputs
            action_tokens:    (B, T-1, num_act_tokens, action_dim) — Qwen-VL action tokens
            predicted_states: (B, T-1, patches, state_dim) — JEPA Predictor outputs

        Returns:
            context: (B, num_output_tokens, output_dim)
        """
        B, T, _, _ = observed_states.shape
        device = observed_states.device

        s = self.state_proj(observed_states.mean(dim=2))       # (B, T, d_model)
        a = self.action_proj(action_tokens.mean(dim=2))        # (B, T-1, d_model)
        s_hat = self.pred_state_proj(predicted_states.mean(dim=2))  # (B, T-1, d_model)

        type_obs = self.token_type_embed(torch.tensor(self.TOKEN_TYPE_OBSERVED, device=device))
        type_act = self.token_type_embed(torch.tensor(self.TOKEN_TYPE_ACTION, device=device))
        type_pred = self.token_type_embed(torch.tensor(self.TOKEN_TYPE_PREDICTED, device=device))
        type_read = self.token_type_embed(torch.tensor(self.TOKEN_TYPE_READOUT, device=device))

        tokens = []
        for t in range(T):
            tokens.append(s[:, t:t+1, :] + type_obs)
            if t < T - 1:
                tokens.append(a[:, t:t+1, :] + type_act)
                tokens.append(s_hat[:, t:t+1, :] + type_pred)

        readout = self.readout_tokens.unsqueeze(0).expand(B, -1, -1) + type_read
        tokens.append(readout)

        x = torch.cat(tokens, dim=1)  # (B, seq_len, d_model)

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        context = x[:, -self.num_output_tokens:, :]
        context = self.output_norm(self.output_proj(context))

        return context
