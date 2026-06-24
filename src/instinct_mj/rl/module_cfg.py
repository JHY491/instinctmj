"""Instinct-RL module configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InstinctRlParallelBlockCfg:
    """Configuration for an encoder parallel block."""

    class_name: str | None = None
    component_names: list[str] = field(default_factory=list)
    output_size: int | None = None
    takeout_input_components: bool = True


@dataclass
class InstinctRlMlpCfg(InstinctRlParallelBlockCfg):
    """Configuration for an MLP encoder block."""

    class_name: str = "MlpModel"
    hidden_sizes: list[int] = field(default_factory=list)
    nonlinearity: str | None = None


@dataclass
class InstinctRlConv2dHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for a Conv2d encoder block."""

    class_name: str = "Conv2dHeadModel"
    channels: list[int] = field(default_factory=list)
    kernel_sizes: list[int] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)
    hidden_sizes: list[int] = field(default_factory=list)
    paddings: list[int] = field(default_factory=list)
    nonlinearity: str | None = None
    use_maxpool: bool = False


@dataclass
class InstinctRlTransformerHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for a Transformer encoder block."""

    class_name: str = "TransformerHeadModel"
    num_heads: int = 4
    num_layers: int = 1
    d_model: int = 256
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "relu"
    nonlinearity: str = "ReLU"
    layer_norm_eps: float = 1.0e-5
    batch_first: bool = True
    norm_first: bool = False
    mask_from_input_dim: int = -1
    output_selection: Literal["maxpool", "smallest_positive", "smallest_nonnegative"] = "maxpool"
    input_hidden_sizes: list[int] = field(default_factory=list)
    output_hidden_sizes: list[int] = field(default_factory=list)


@dataclass
class InstinctRlPerceptiveEncoderCfg:
    """Configuration block for PerceptiveEncoder (depth + dynamics fusion)."""

    class_name: str = "PerceptiveEncoder"
    depth_component_names: list[str] = field(default_factory=lambda: ["depth_image"])
    dynamics_component_names: list[str] = field(default_factory=list)
    depth_channels: list[int] = field(default_factory=lambda: [32, 64])
    depth_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    depth_strides: list[int] = field(default_factory=lambda: [2, 2])
    depth_paddings: list[int] = field(default_factory=lambda: [1, 1])
    film_hidden_dim: int = 64
    terrain_mlp_hidden: list[int] = field(default_factory=lambda: [128])
    dynamics_mlp_hidden: list[int] = field(default_factory=lambda: [256, 128])
    output_size: int = 256
    nonlinearity: str = "ReLU"
    takeout_input_components: bool = True


__all__ = [
    "InstinctRlParallelBlockCfg",
    "InstinctRlMlpCfg",
    "InstinctRlConv2dHeadCfg",
    "InstinctRlTransformerHeadCfg",
    "InstinctRlPerceptiveEncoderCfg",
]
