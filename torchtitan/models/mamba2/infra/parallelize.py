# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file applies parallelism to the Mamba2 model.

For initial implementation, we focus on DDP only using replicate().
"""

import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_mamba2(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply DDP to Mamba2 using replicate().

    Args:
        model: Mamba2 model
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration
    """
    world_mesh = parallel_dims.world_mesh

    # Warn if unsupported parallelism is requested
    if parallel_dims.tp_enabled:
        logger.warning("Tensor Parallelism not supported for Mamba2. Ignoring.")
    if parallel_dims.cp_enabled:
        logger.warning("Context Parallelism not supported for Mamba2. Ignoring.")
    if parallel_dims.pp_enabled:
        logger.warning("Pipeline Parallelism not supported for Mamba2. Ignoring.")
    if parallel_dims.fsdp_enabled:
        logger.warning("FSDP not supported for Mamba2 yet. Using DDP instead.")

    # Apply DDP via replicate
    if parallel_dims.dp_enabled:
        replicate(model, device_mesh=world_mesh, bucket_cap_mb=100)
        logger.info("Applied DDP to the Mamba2 model")

    return model
