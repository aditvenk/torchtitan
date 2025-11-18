# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mamba2 model integration for TorchTitan.

This module provides Mamba2 model configurations and training specifications,
reusing existing TorchTitan components for tokenization, data loading, and optimization.
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_mamba2
from .model.args import Mamba2ModelArgs
from .model.model import Mamba2

__all__ = [
    "parallelize_mamba2",
    "Mamba2ModelArgs",
    "Mamba2",
    "mamba2_args",
]


# Mamba2 model configurations
mamba2_args = {
    # Debug model: ~3M parameters for quick testing
    "debugmodel": Mamba2ModelArgs(
        d_model=256,
        n_layers=4,
        vocab_size=2048,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        max_seq_len=2048,
    ),
    # Small model: ~30M parameters - intermediate size for testing
    "30M": Mamba2ModelArgs(
        d_model=512,
        n_layers=8,
        vocab_size=50280,
        d_state=64,  # Smaller state than 130M
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        max_seq_len=2048,
    ),
    # Medium model: ~130M parameters (may need shorter sequences)
    "130M": Mamba2ModelArgs(
        d_model=768,
        n_layers=16,
        vocab_size=50280,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        max_seq_len=2048,
    ),
    # Large model: ~350M parameters
    "370M": Mamba2ModelArgs(
        d_model=1024,
        n_layers=24,
        vocab_size=50280,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        max_seq_len=2048,
    ),
    # Large model: ~1.4B parameters (matches state-spaces/mamba)
    "1.4B": Mamba2ModelArgs(
        d_model=1536,
        n_layers=48,
        vocab_size=50280,
        d_state=128,  # Standard for Mamba2 large models
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        max_seq_len=2048,
    ),
}


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for Mamba2.

    This reuses existing TorchTitan components:
    - Tokenizer: HuggingFace tokenizer (same as Llama)
    - Data: Text dataloader with C4 dataset
    - Loss: Cross-entropy loss
    - Optimizer: AdamW
    - LR Scheduler: Linear warmup + decay
    """
    return TrainSpec(
        model_cls=Mamba2,
        model_args=mamba2_args,
        parallelize_fn=parallelize_mamba2,
        pipelining_fn=pipeline_llm,  # Basic pipelining support
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,  # Optional: can add later for HF compatibility
    )
