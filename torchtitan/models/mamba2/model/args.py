# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Mamba2ModelArgs(BaseModelArgs):
    """Arguments for Mamba2 model configuration."""

    d_model: int = 768  # Model dimension
    n_layers: int = 24  # Number of Mamba2 blocks
    vocab_size: int = 50280  # Vocabulary size

    # SSM specific parameters
    d_state: int = 128  # SSM state dimension (N in paper)
    d_conv: int = 4  # Local convolution width
    expand: int = 2  # Expansion factor for inner dimension

    # Head dimension for Mamba2
    headdim: int = 64  # Head dimension (new in Mamba2)
    ngroups: int = 1  # Number of groups for GQA-like structure

    # Additional parameters
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    pad_vocab_size_multiple: int = 8  # Pad vocab size to multiple of this

    # Training specific
    depth_init: bool = True  # Use depth-based initialization

    def __post_init__(self):
        # Ensure vocab_size is padded to multiple
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size = (
                (self.vocab_size // self.pad_vocab_size_multiple + 1)
                * self.pad_vocab_size_multiple
            )

        # Ensure d_model is divisible by headdim * ngroups
        assert self.d_model % (self.headdim * self.ngroups) == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"headdim * ngroups ({self.headdim * self.ngroups})"
        )

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model args from training config."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        """Calculate number of parameters and FLOPs."""
        # Count parameters
        nparams = sum(p.numel() for p in model.parameters())

        # Rough FLOP estimate for Mamba2
        # Each layer has:
        # - Projections: ~6 * d_model * d_inner (in_proj, out_proj, conv, A, B, C, D)
        # - Convolution: d_conv * d_inner
        # - SSM scan: 2 * seq_len * d_state * d_inner (simplified)

        d_inner = self.expand * self.d_model

        # Per-token FLOPs per layer (approximate)
        projection_flops = 6 * self.d_model * d_inner
        conv_flops = self.d_conv * d_inner
        ssm_flops = 2 * self.d_state * d_inner

        flops_per_token_per_layer = projection_flops + conv_flops + ssm_flops

        # Total FLOPs for forward pass
        total_flops = (
            2 * seq_len * self.n_layers * flops_per_token_per_layer
        )

        # Account for embedding and output layers
        total_flops += 2 * seq_len * self.d_model * self.vocab_size

        return nparams, total_flops
