# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mamba2 model implementation using plain PyTorch operations.
Based on the Mamba2 architecture from https://github.com/state-spaces/mamba
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.model import ModelProtocol

from .args import Mamba2ModelArgs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def selective_scan_seq(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Memory-efficient plain PyTorch implementation of selective scan (sequential).

    This implementation computes discretization step-by-step to avoid
    materializing large tensors for the entire sequence.

    Args:
        x: Input tensor (batch, seq_len, d_inner)
        delta: Time-step parameter (batch, seq_len, d_inner)
        A: State transition parameter (d_inner, d_state)
        B: Input projection (batch, seq_len, ngroups, d_state)
        C: Output projection (batch, seq_len, ngroups, d_state)
        D: Skip connection parameter (d_inner,)

    Returns:
        Output tensor (batch, seq_len, d_inner)
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]
    ngroups = B.shape[2]
    heads_per_group = d_inner // ngroups

    # Reshape x to (batch, seq_len, ngroups, heads_per_group)
    x_grouped = x.reshape(batch, seq_len, ngroups, heads_per_group)

    # Reshape delta_t for group structure
    delta_grouped = delta.reshape(batch, seq_len, ngroups, heads_per_group)

    # Reshape A for efficient computation
    A_reshaped = A.reshape(ngroups, heads_per_group, d_state)

    # Initialize state
    state = torch.zeros(batch, ngroups, heads_per_group, d_state, device=x.device, dtype=x.dtype)

    # Pre-allocate output tensor instead of using list
    y = torch.empty(batch, seq_len, ngroups, heads_per_group, device=x.device, dtype=x.dtype)

    # Sequential scan - compute discretization per timestep to save memory
    for t in range(seq_len):
        # Get tensors at timestep t
        delta_t = delta_grouped[:, t]  # (batch, ngroups, heads_per_group)
        B_t = B[:, t]  # (batch, ngroups, d_state)
        C_t = C[:, t]  # (batch, ngroups, d_state)
        x_t = x_grouped[:, t]  # (batch, ngroups, heads_per_group)

        # Discretization for timestep t only
        # A_bar = exp(delta * A) - compute only for this timestep
        A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))

        # B_bar = delta * B
        B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)  # (batch, ngroups, heads_per_group, d_state)

        # Update state: state = A_bar * state + B_bar * x
        state = A_bar_t * state + B_bar_t * x_t.unsqueeze(-1)

        # Compute output: y = C * state
        # Write directly to pre-allocated tensor
        y[:, t] = torch.einsum('bghd,bgd->bgh', state, C_t)

    # Reshape output
    y = y.reshape(batch, seq_len, d_inner)

    # Add skip connection
    if D is not None:
        y = y + x * D

    return y


class SelectiveScanFunction(torch.autograd.Function):
    """
    Custom autograd function for selective scan with memory-efficient backward pass.

    This implementation uses reverse-mode scan to compute gradients without storing
    all intermediate states in the computation graph, significantly reducing memory usage.
    """

    @staticmethod
    def forward(ctx, x, delta, A, B, C, D, ngroups):
        """
        Forward pass: compute selective scan and save only inputs.

        Args:
            ctx: Context for saving tensors
            x: Input tensor (batch, seq_len, d_inner)
            delta: Time-step parameter (batch, seq_len, d_inner)
            A: State transition parameter (d_inner, d_state)
            B: Input projection (batch, seq_len, ngroups, d_state)
            C: Output projection (batch, seq_len, ngroups, d_state)
            D: Skip connection parameter (d_inner,)
            ngroups: Number of groups

        Returns:
            Output tensor (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        heads_per_group = d_inner // ngroups

        # Reshape inputs
        x_grouped = x.reshape(batch, seq_len, ngroups, heads_per_group)
        delta_grouped = delta.reshape(batch, seq_len, ngroups, heads_per_group)
        A_reshaped = A.reshape(ngroups, heads_per_group, d_state)

        # Initialize state
        state = torch.zeros(batch, ngroups, heads_per_group, d_state, device=x.device, dtype=x.dtype)

        # Pre-allocate output
        y = torch.empty(batch, seq_len, ngroups, heads_per_group, device=x.device, dtype=x.dtype)

        # Forward scan
        for t in range(seq_len):
            delta_t = delta_grouped[:, t]
            B_t = B[:, t]
            C_t = C[:, t]
            x_t = x_grouped[:, t]

            # Compute discretization
            A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))
            B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)

            # Update state
            state = A_bar_t * state + B_bar_t * x_t.unsqueeze(-1)

            # Compute output
            y[:, t] = torch.einsum('bghd,bgd->bgh', state, C_t)

        # Reshape output
        y = y.reshape(batch, seq_len, d_inner)

        # Add skip connection
        if D is not None:
            y = y + x * D

        # Save for backward - only save inputs, not intermediate states
        ctx.save_for_backward(x, delta, A, B, C, D)
        ctx.ngroups = ngroups

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: recompute states and compute gradients using reverse-mode scan.

        This recomputes states on-the-fly during backward pass instead of storing them
        in the computation graph, trading computation for memory.
        """
        x, delta, A, B, C, D = ctx.saved_tensors
        ngroups = ctx.ngroups

        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        heads_per_group = d_inner // ngroups

        # Reshape inputs
        x_grouped = x.reshape(batch, seq_len, ngroups, heads_per_group)
        delta_grouped = delta.reshape(batch, seq_len, ngroups, heads_per_group)
        A_reshaped = A.reshape(ngroups, heads_per_group, d_state)

        # Handle skip connection gradient
        grad_D = None
        if D is not None:
            grad_D = (grad_output * x).sum(dim=(0, 1))
            grad_y = grad_output.clone()
        else:
            grad_y = grad_output

        grad_y_grouped = grad_y.reshape(batch, seq_len, ngroups, heads_per_group)

        # Initialize gradients
        grad_x = torch.zeros_like(x_grouped)
        grad_delta = torch.zeros_like(delta_grouped)
        grad_A = torch.zeros_like(A_reshaped)
        grad_B = torch.zeros_like(B)
        grad_C = torch.zeros_like(C)

        # Recompute all states (forward pass)
        # We need these to compute gradients, but we don't store them in the graph
        states = []
        state = torch.zeros(batch, ngroups, heads_per_group, d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            delta_t = delta_grouped[:, t]
            B_t = B[:, t]
            x_t = x_grouped[:, t]

            A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))
            B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)

            state = A_bar_t * state + B_bar_t * x_t.unsqueeze(-1)
            states.append(state.clone())

        # Backward scan through time
        grad_state = torch.zeros_like(state)

        for t in range(seq_len - 1, -1, -1):
            delta_t = delta_grouped[:, t]  # (batch, ngroups, heads_per_group)
            B_t = B[:, t]  # (batch, ngroups, d_state)
            C_t = C[:, t]  # (batch, ngroups, d_state)
            x_t = x_grouped[:, t]  # (batch, ngroups, heads_per_group)

            A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))  # (batch, ngroups, heads_per_group, d_state)
            B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)  # (batch, ngroups, heads_per_group, d_state)

            state_t = states[t]
            state_prev = states[t - 1] if t > 0 else torch.zeros_like(state_t)

            # Gradient from output: y[:, t] = einsum('bghd,bgd->bgh', state_t, C_t)
            grad_state_from_output = torch.einsum('bgh,bgd->bghd', grad_y_grouped[:, t], C_t)
            grad_C[:, t] = torch.einsum('bghd,bgh->bgd', state_t, grad_y_grouped[:, t])

            # Total gradient w.r.t state at timestep t
            grad_state_t = grad_state_from_output + grad_state

            # Gradient w.r.t. state update: state_t = A_bar_t * state_prev + B_bar_t * x_t.unsqueeze(-1)
            # Gradient flows back to previous state
            grad_state = A_bar_t * grad_state_t

            # Gradient w.r.t. A_bar_t: from state_t = A_bar_t * state_prev + ...
            grad_A_bar_t = state_prev * grad_state_t

            # Gradient w.r.t. B_bar_t: from state_t = ... + B_bar_t * x_t.unsqueeze(-1)
            grad_B_bar_t = x_t.unsqueeze(-1) * grad_state_t

            # Gradient w.r.t. x_t: from state_t = ... + B_bar_t * x_t.unsqueeze(-1)
            grad_x[:, t] = (B_bar_t * grad_state_t).sum(dim=-1)

            # Gradient w.r.t. B_t: from B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)
            # Shape: B_bar_t is (batch, ngroups, heads_per_group, d_state)
            # B_t.unsqueeze(2) expands (batch, ngroups, d_state) -> (batch, ngroups, 1, d_state)
            # delta_t.unsqueeze(-1) expands (batch, ngroups, heads_per_group) -> (batch, ngroups, heads_per_group, 1)
            # So B_bar_t[b,g,h,d] = delta_t[b,g,h] * B_t[b,g,d]
            grad_B[:, t] = (grad_B_bar_t * delta_t.unsqueeze(-1)).sum(dim=2)

            # Gradient w.r.t. delta_t (affects both A_bar_t and B_bar_t)
            # From A_bar_t = exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))
            # A_reshaped is (ngroups, heads_per_group, d_state)
            # A_bar_t[b,g,h,d] = exp(delta_t[b,g,h] * A_reshaped[g,h,d])
            # dA_bar_t/d(delta_t[b,g,h]) = A_bar_t[b,g,h,d] * A_reshaped[g,h,d]
            grad_delta_from_A = (grad_A_bar_t * A_bar_t * A_reshaped.unsqueeze(0)).sum(dim=-1)

            # From B_bar_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)
            # B_bar_t[b,g,h,d] = delta_t[b,g,h] * B_t[b,g,d]
            # dB_bar_t/d(delta_t[b,g,h]) = B_t[b,g,d]
            grad_delta_from_B = (grad_B_bar_t * B_t.unsqueeze(2)).sum(dim=-1)

            grad_delta[:, t] = grad_delta_from_A + grad_delta_from_B

            # Gradient w.r.t. A: from A_bar_t = exp(delta_t.unsqueeze(-1) * A_reshaped.unsqueeze(0))
            # A_bar_t[b,g,h,d] = exp(delta_t[b,g,h] * A_reshaped[g,h,d])
            # dA_bar_t/dA_reshaped[g,h,d] = A_bar_t[b,g,h,d] * delta_t[b,g,h]
            grad_A += (grad_A_bar_t * A_bar_t * delta_t.unsqueeze(-1)).sum(dim=0)

        # Reshape gradients
        grad_x = grad_x.reshape(batch, seq_len, d_inner)
        grad_delta = grad_delta.reshape(batch, seq_len, d_inner)
        grad_A = grad_A.reshape(d_inner, d_state)

        # Add gradient from skip connection: y = y_ssm + x * D
        # So d(y)/d(x) includes D term
        if D is not None:
            grad_x = grad_x + grad_output * D

        # Return gradients for all inputs (x, delta, A, B, C, D, ngroups)
        # ngroups is not a tensor, so return None for its gradient
        return grad_x, grad_delta, grad_A, grad_B, grad_C, grad_D, None


def selective_scan_efficient(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Memory-efficient selective scan using custom autograd function.

    This uses reverse-mode scan in the backward pass to avoid storing all
    intermediate states in the computation graph. While this still recomputes
    states during backward, they are not saved in PyTorch's autograd graph,
    significantly reducing memory usage.

    Args:
        x: Input tensor (batch, seq_len, d_inner)
        delta: Time-step parameter (batch, seq_len, d_inner)
        A: State transition parameter (d_inner, d_state)
        B: Input projection (batch, seq_len, ngroups, d_state)
        C: Output projection (batch, seq_len, ngroups, d_state)
        D: Skip connection parameter (d_inner,)

    Returns:
        Output tensor (batch, seq_len, d_inner)
    """
    ngroups = B.shape[2]
    return SelectiveScanFunction.apply(x, delta, A, B, C, D, ngroups)


class Mamba2Block(nn.Module):
    """Single Mamba2 block with SSM, convolution, and projections."""

    def __init__(self, args: Mamba2ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        self.d_model = args.d_model
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.d_inner = self.expand * self.d_model

        self.headdim = args.headdim
        self.ngroups = args.ngroups
        self.nheads = self.d_inner // self.headdim

        # Input projection: x -> (x_proj, z) for gating
        # x: (B, L, d_model) --> (B, L, d_inner * 2)
        # x_proj, z: (B, L, d_inner)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,  # Depthwise conv
            padding=self.d_conv - 1,
            bias=True,
        )

        # SSM parameters
        # A: State transition matrix (d_inner, d_state)
        # Initialized to -0.5 to 0.5 range, will be learned
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.5)

        # D: Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Projections for B, C, delta_t
        self.x_proj = nn.Linear(self.d_inner, self.d_state * self.ngroups * 2 + self.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Normalization
        self.norm = RMSNorm(self.d_model, eps=args.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Normalize
        residual = x
        x = self.norm(x)

        # Input projection with gating
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each is (batch, seq_len, d_inner)

        # 1D convolution (depthwise)
        # Conv1d expects (batch, channels, seq_len)
        # The slice :seq_len is to remove the outputs produced due to padding
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)  # (batch, seq_len, d_inner)

        # Activation
        x_conv = F.silu(x_conv)

        # Project to get B, C, delta_t
        x_proj = self.x_proj(x_conv)  # (batch, seq_len, d_state * ngroups * 2 + d_inner)

        # Split into B, C, delta_t
        split_sizes = [self.d_state * self.ngroups, self.d_state * self.ngroups, self.d_inner]
        B, C, delta = torch.split(x_proj, split_sizes, dim=-1)

        # Reshape B and C to (batch, seq_len, ngroups, d_state)
        B = B.reshape(batch, seq_len, self.ngroups, self.d_state)
        C = C.reshape(batch, seq_len, self.ngroups, self.d_state)

        # delta_t needs to be positive (softplus ensures this)
        delta = F.softplus(delta)

        # Selective scan (the core SSM operation)
        # Use efficient scan with custom autograd - memory efficient for long sequences
        y = selective_scan_seq(x_conv, delta, self.A, B, C, self.D)

        # Gating with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Residual connection
        output = output + residual

        return output

    def init_weights(self):
        """Initialize weights for this block."""
        # Initialize projections
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.02)

        # Initialize A with negative values for stability
        nn.init.uniform_(self.A, -0.5, -0.1)

        # Initialize D
        nn.init.ones_(self.D)

        # Conv weights
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)


class Mamba2(nn.Module, ModelProtocol):
    """
    Mamba2 language model.

    Args:
        args: Model configuration
    """

    def __init__(self, args: Mamba2ModelArgs):
        super().__init__()
        self.model_args = args  # Store as model_args for TorchTitan compatibility

        # Token embeddings (using TorchTitan naming convention)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.d_model)

        # Mamba2 blocks - using ModuleList instead of ModuleDict for DDP compatibility
        self.layers = nn.ModuleList([
            Mamba2Block(args, layer_idx=i) for i in range(args.n_layers)
        ])

        # Final normalization (using TorchTitan naming convention)
        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)

        # Output projection to vocabulary (using TorchTitan naming convention)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # Tie weights between embedding and output projection (matching mamba-ssm repo)
        self.output.weight = self.tok_embeddings.weight

        # For TorchTitan compatibility: Mamba2 doesn't use rotary position embeddings
        # Set freqs_cis to None (required by training loop context parallel setup)
        self.freqs_cis = None

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: Input token IDs (batch, seq_len)
            input_pos: Optional position indices (not used in Mamba2, kept for interface compatibility)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.tok_embeddings(tokens)

        # Apply Mamba2 blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output(x)

        return logits

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        Initialize all weights in the model.

        Args:
            buffer_device: Optional device for buffers (kept for API compatibility,
                          not used in Mamba2 as it has no persistent buffers like freqs_cis)
        """
        # Initialize embeddings
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        # Initialize each layer
        for layer in self.layers:
            if layer is not None:
                layer.init_weights()

        # Initialize final norm
        if self.norm is not None:
            if hasattr(self.norm, 'reset_parameters'):
                self.norm.reset_parameters()

        # Initialize output projection
        # Use similar initialization as Llama3 for the output layer
        if self.output is not None:
            final_out_std = self.model_args.d_model**-0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def reset_parameters(self):
        """Reset parameters (called by some training frameworks)."""
        self.init_weights()
