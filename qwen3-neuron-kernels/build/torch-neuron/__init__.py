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
from nkilib.core.output_projection.output_projection_cte.output_projection_cte import (
    output_projection_cte as nki_output_projection_cte,
)
from nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8 import mlp_forward_mxfp8_nki

_mlp_forward_mxfp8_nki = nki.jit(mlp_forward_mxfp8_nki)


def _rotate_half(x):
    """Rotate half the head dimension (GPT-NeoX / Qwen3 convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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

        D = self.head_dim
        N_q = self.num_heads
        N_kv = self.num_kv_heads

        # NOTE: We deliberately do NOT use nki_qkv here. Its prefill path
        # (qkv_cte, selected for seqlen > SEQLEN_THRESHOLD_FOR_QKV_CTE == 96)
        # is broken in this Neuron build: it does not write its output buffer,
        # returning near-zero / non-deterministic garbage that then overflows
        # to NaN downstream. The QKV projection is a cheap GEMM, so we compute
        # it (plus QK-norm and RoPE) in torch and feed the result to the
        # attention_cte kernel, which is correct and stable.
        qkv = torch.matmul(hidden_states, self.fused_qkv_proj.weight.T)  # [B, S, (N_q+2*N_kv)*D]
        q_size = N_q * D
        kv_size = N_kv * D
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(B, S, N_q, D)
        k = k.view(B, S, N_kv, D)
        v = v.view(B, S, N_kv, D)

        # Qwen3 RMS QK-norm over the head dimension (computed in fp32).
        in_dtype = hidden_states.dtype
        qf = q.float()
        kf = k.float()
        q = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps) * self.q_norm).to(in_dtype)
        k = (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps) * self.k_norm).to(in_dtype)

        # [B, S, N, D] -> [B, N, S, D] and apply RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos_u = cos.unsqueeze(1)  # [B, 1, S, D]
        sin_u = sin.unsqueeze(1)
        q = q * cos_u + _rotate_half(q) * sin_u
        k = k * cos_u + _rotate_half(k) * sin_u

        # Reshape to attention_cte layout: [B*N, S, D]
        q = q.reshape(B * N_q, S, D).contiguous()  # [B*N_q,  S, D]
        k = k.reshape(B * N_kv, S, D).contiguous()  # [B*N_kv, S, D]
        v = v.reshape(B * N_kv, S, D).contiguous()  # [B*N_kv, S, D]

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

        # Output projection via NKI (output_projection_cte is correct & stable):
        # reshape to [B, N_q, D, S] as the kernel expects.
        attn_out = attn_out.reshape(B, N_q, S, D)
        attn_out = attn_out.permute(0, 1, 3, 2)  # [B, N_q, D, S]
        output = nki_output_projection_cte(attn_out, self.o_proj.weight.T)

        return output, None


class layers:
    NeuronRMSNormMLP = NeuronRMSNormMLP
    NeuronMLPMXFP8 = NeuronMLPMXFP8
    NeuronQwen3Attention = NeuronQwen3Attention
