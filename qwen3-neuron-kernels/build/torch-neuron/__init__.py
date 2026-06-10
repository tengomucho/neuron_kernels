import torch
import torch.nn as nn

from transformers import Concatenate, WeightConverter, WeightRenaiming

from nkilib.core.mlp import mlp as nki_mlp
from nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType


class NeuronRMSNormMLPLayout(nn.Module):
    conversion_mapping = [
        WeightRenaiming(
            source_patterns=r"model.layers.(\d+).post_attention_layernorm.weight",
            target_patterns=r"model.layers.\1.post_attention_layernorm.norm_weight",
        ),
        WeightRenaiming(
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
        self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation function: {config.hidden_act}. Only 'silu' is supported.")

        self.activation_fn = ActFnType.SiLU


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
        )[0] # nki_mlp returns a list, the first element is the output tensor.


class layers:
    NeuronRMSNormMLP = NeuronRMSNormMLP
