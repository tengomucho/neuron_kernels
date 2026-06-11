import torch
import torch.nn as nn
import nki
import nki.language as nl

from transformers.core_model_loading import WeightRenaming, WeightConverter, Concatenate

from nkilib.core.mlp.mlp import mlp as nki_mlp
from nkilib.core.utils.common_types import (
    ActFnType,
    NormType,
    QuantizationType,
    QKNormConfig,
    QKVOutputLayout,
)
from nkilib.core.qkv.qkv import qkv as nki_qkv
from nkilib.core.attention.attention_cte import attention_cte as nki_attention_cte
from nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8 import mlp_forward_mxfp8_nki

_mlp_forward_mxfp8_nki = nki.jit(mlp_forward_mxfp8_nki)


class NeuronRMSNormMLPLayout(nn.Module):
    conversion_mapping = [
        WeightRenaming(
            source_patterns=r"model.layers.(\d+).post_attention_layernorm.weight",
            target_patterns=r"model.layers.\1.post_attention_layernorm.norm_weight",
        ),
        WeightRenaming(
            source_patterns=r"model.layers.(\d+).mlp.(gate|up|down)_proj.weight",
            target_patterns=r"model.layers.\1.post_attention_layernorm.\2_proj.weight",
        ),
    ]

    def __init__(self, norm, mlp):
        super().__init__()

        # Norm attributes
        self.norm_weight = nn.Parameter(torch.ones_like(norm.weight))
        self.variance_epsilon = norm.variance_epsilon

        # MLP attributes
        config = mlp.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation function: {config.hidden_act}. Only 'silu' is supported."
            )

        self.activation_fn = ActFnType.SiLU

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass


class NeuronRMSNormMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nki_mlp(
            hidden_states,
            gate_proj_weights_tensor=self.gate_proj.weight.T,
            up_proj_weights_tensor=self.up_proj.weight.T,
            down_proj_weights_tensor=self.down_proj.weight.T,
            normalization_weights_tensor=self.norm_weight.unsqueeze(0),
            activation_fn=self.activation_fn,
            normalization_type=NormType.RMS_NORM,
            quantization_type=QuantizationType.NONE,
            eps=self.variance_epsilon,
        )[0]  # nki_mlp returns a list, the first element is the output tensor.


class NeuronMLPMXFP8Layout(nn.Module):
    conversion_mapping = [
        WeightConverter(
            source_patterns=[r".mlp.gate_proj", r".mlp.up_proj"],
            target_patterns=r".mlp.gate_up_proj",
            operations=[Concatenate(dim=0)],
        ),
    ]

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass


class NeuronMLPMXFP8(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        intermediate = torch.empty(
            (batch_size * seq_length, self.intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        output = _mlp_forward_mxfp8_nki(
            hidden_states,
            self.gate_up_proj.weight,
            self.down_proj.weight,
            intermediate,
            run_with_lnc2=True,
            dtype=nl.bfloat16,
        )

        return output.view(batch_size, seq_length, hidden_dim)


class NeuronQwen3AttentionLayout(nn.Module):
    conversion_mapping = [
        WeightConverter(
            source_patterns=[
                ".self_attn.q_proj",
                ".self_attn.k_proj",
                ".self_attn.v_proj",
            ],
            target_patterns=r".self_attn.fused_qkv_proj",
            operations=[Concatenate(dim=0)],
        ),
        WeightRenaming(
            source_patterns=r"model.layers.(\d+).self_attn.q_norm.weight",
            target_patterns=r"model.layers.\1.self_attn.q_norm",
        ),
        WeightRenaming(
            source_patterns=r"model.layers.(\d+).self_attn.k_norm.weight",
            target_patterns=r"model.layers.\1.self_attn.k_norm",
        ),
    ]

    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        H = config.hidden_size
        D = config.head_dim
        N_q = config.num_attention_heads
        N_kv = config.num_key_value_heads

        self.fused_qkv_proj = nn.Linear(H, (N_q + 2 * N_kv) * D, bias=False)
        self.o_proj = nn.Linear(N_q * D, H, bias=False)

        self.q_norm = nn.Parameter(torch.ones(D))
        self.k_norm = nn.Parameter(torch.ones(D))

        self.num_heads = N_q
        self.num_kv_heads = N_kv
        self.head_dim = D
        self.qk_norm_eps = config.rms_norm_eps

        self.qk_norm_cfg = QKNormConfig(
            q_norm=NormType.RMS_NORM,
            k_norm=NormType.RMS_NORM,
            eps=self.qk_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ):
        pass


class NeuronQwen3Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        **kwargs,
    ):
        B, S, H = hidden_states.shape
        cos, sin = position_embeddings  # each [B, S, D]

        # Fused QKV projection + per-head QK-norm + RoPE
        qkv_out = nki_qkv(
            hidden_states,
            self.fused_qkv_proj.weight.T,
            output_layout=QKVOutputLayout.BSD,
            fused_norm_type=NormType.NO_NORM,
            fused_rope=True,
            cos_cache=cos,
            sin_cache=sin,
            d_head=self.head_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            qk_norm_pre_rope=self.qk_norm_cfg,
            qk_norm_pre_rope_q_gamma=self.q_norm.unsqueeze(0),  # [1, D]
            qk_norm_pre_rope_k_gamma=self.k_norm.unsqueeze(0),  # [1, D]
        )  # [B, S, (N_q + 2*N_kv) * D]

        # Split and reshape for attention_cte: [B*N, S, D]
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
        q = q.reshape(B * self.num_heads, S, self.head_dim)  # [B*N_q,  S, D]
        k = k.reshape(B * self.num_kv_heads, S, self.head_dim)  # [B*N_kv, S, D]
        v = v.reshape(B * self.num_kv_heads, S, self.head_dim)  # [B*N_kv, S, D]

        # Flash attention (causal, GQA via batch_size_q vs batch_size_kv)
        attn_out = nki_attention_cte(
            q,
            k,
            v,
            scale=self.head_dim**-0.5,
            causal_mask=True,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )  # [B*N_q, S, D]

        # Output projection
        attn_out = attn_out.view(B, self.num_heads, S, self.head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B, S, N_q*D]
        output = self.o_proj(attn_out)

        return output, None


class layers:
    NeuronRMSNormMLP = NeuronRMSNormMLP
    NeuronMLPMXFP8 = NeuronMLPMXFP8
    NeuronQwen3Attention = NeuronQwen3Attention
