# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Mamba2 selective scan implementations.

Tests verify that different implementations (sequential, efficient)
produce the same outputs given the same inputs.
"""

import pytest
import torch

from torchtitan.models.mamba2.model.model import (
    selective_scan_efficient,
    selective_scan_seq,
)


class TestSelectiveScan:
    """Test suite for selective scan implementations."""

    @pytest.fixture
    def small_inputs(self):
        """Create small test inputs for quick tests."""
        batch = 2
        seq_len = 64
        d_inner = 128
        d_state = 16
        ngroups = 1

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, d_inner)
        A = torch.randn(d_inner, d_state) * 0.5
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        return x, delta, A, B, C, D

    @pytest.fixture
    def medium_inputs(self):
        """Create medium-sized test inputs."""
        batch = 4
        seq_len = 256
        d_inner = 512
        d_state = 64
        ngroups = 1

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, d_inner)
        A = torch.randn(d_inner, d_state) * 0.5
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        return x, delta, A, B, C, D

    @pytest.fixture
    def large_inputs(self):
        """Create larger test inputs (closer to real model size)."""
        batch = 2
        seq_len = 512
        d_inner = 1024
        d_state = 128
        ngroups = 1

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, d_inner)
        A = torch.randn(d_inner, d_state) * 0.5
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        return x, delta, A, B, C, D

    def test_selective_scan_seq_shape(self, small_inputs):
        """Test that sequential scan produces correct output shape."""
        x, delta, A, B, C, D = small_inputs
        batch, seq_len, d_inner = x.shape

        y = selective_scan_seq(x, delta, A, B, C, D)

        assert y.shape == (batch, seq_len, d_inner), \
            f"Expected shape {(batch, seq_len, d_inner)}, got {y.shape}"

    def test_skip_connection_none(self, small_inputs):
        """Test that scans work correctly when D (skip connection) is None."""
        x, delta, A, B, C, _ = small_inputs

        y_seq = selective_scan_seq(x, delta, A, B, C, D=None)

        # Just verify it runs without error and has correct shape
        assert y_seq.shape == x.shape

    @pytest.mark.parametrize("ngroups", [1, 2, 4])
    def test_different_ngroups(self, ngroups):
        """Test selective scan with different numbers of groups."""
        batch = 2
        seq_len = 64
        d_inner = 128
        d_state = 16

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, d_inner)
        A = torch.randn(d_inner, d_state) * 0.5
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        y_seq = selective_scan_seq(x, delta, A, B, C, D)

        # Just verify it runs without error and has correct shape
        assert y_seq.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_selective_scan_cuda(self, small_inputs):
        """Test selective scan on CUDA if available."""
        x, delta, A, B, C, D = small_inputs

        # Move to CUDA
        x_cuda = x.cuda()
        delta_cuda = delta.cuda()
        A_cuda = A.cuda()
        B_cuda = B.cuda()
        C_cuda = C.cuda()
        D_cuda = D.cuda()

        # Compute on CPU
        y_cpu = selective_scan_seq(x, delta, A, B, C, D)

        # Compute on CUDA
        y_cuda = selective_scan_seq(x_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D_cuda)

        # Compare (move CUDA result back to CPU)
        torch.testing.assert_close(
            y_cpu, y_cuda.cpu(),
            rtol=1e-4, atol=1e-4,
            msg="CPU and CUDA results should match"
        )

    def test_gradient_flow_seq(self, small_inputs):
        """Test that gradients flow correctly through sequential scan."""
        x, delta, A, B, C, D = small_inputs

        # Require gradients
        x.requires_grad_(True)
        delta.requires_grad_(True)
        A.requires_grad_(True)
        B.requires_grad_(True)
        C.requires_grad_(True)
        D.requires_grad_(True)

        y = selective_scan_seq(x, delta, A, B, C, D)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert delta.grad is not None and not torch.all(delta.grad == 0)
        assert A.grad is not None and not torch.all(A.grad == 0)
        assert B.grad is not None and not torch.all(B.grad == 0)
        assert C.grad is not None and not torch.all(C.grad == 0)
        assert D.grad is not None and not torch.all(D.grad == 0)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch, seq_len, d_inner, d_state, ngroups = 2, 32, 64, 16, 1

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner) * 10  # Large values
        delta = torch.randn(batch, seq_len, d_inner).abs() * 0.1  # Small positive values
        A = torch.randn(d_inner, d_state) * 0.01  # Small A for stability
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        # Check no NaN or Inf
        assert not torch.isnan(y_seq).any(), "Sequential scan produced NaN"
        assert not torch.isinf(y_seq).any(), "Sequential scan produced Inf"
        assert not torch.isnan(y_efficient).any(), "Efficient scan produced NaN"
        assert not torch.isinf(y_efficient).any(), "Efficient scan produced Inf"

        # Should still match
        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-4, atol=1e-4,
            msg="Sequential and efficient scans should match even with extreme values"
        )

    def test_efficient_vs_seq_small(self, small_inputs):
        """Test that efficient and sequential produce same results on small inputs."""
        x, delta, A, B, C, D = small_inputs

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-5, atol=1e-5,
            msg="Sequential and efficient scans should produce same output"
        )

    def test_efficient_vs_seq_medium(self, medium_inputs):
        """Test that efficient and sequential produce same results on medium inputs."""
        x, delta, A, B, C, D = medium_inputs

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-5, atol=1e-5,
            msg="Sequential and efficient scans should produce same output"
        )

    def test_efficient_vs_seq_large(self, large_inputs):
        """Test that efficient and sequential produce same results on large inputs."""
        x, delta, A, B, C, D = large_inputs

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-5, atol=1e-5,
            msg="Sequential and efficient scans should produce same output"
        )

    def test_gradients_match_seq_vs_efficient(self, small_inputs):
        """Test that gradients match between sequential and efficient scans."""
        x, delta, A, B, C, D = small_inputs

        # Sequential gradients
        x_seq = x.clone().requires_grad_(True)
        delta_seq = delta.clone().requires_grad_(True)
        A_seq = A.clone().requires_grad_(True)
        B_seq = B.clone().requires_grad_(True)
        C_seq = C.clone().requires_grad_(True)
        D_seq = D.clone().requires_grad_(True)

        y_seq = selective_scan_seq(x_seq, delta_seq, A_seq, B_seq, C_seq, D_seq)
        loss_seq = y_seq.sum()
        loss_seq.backward()

        # Efficient gradients
        x_eff = x.clone().requires_grad_(True)
        delta_eff = delta.clone().requires_grad_(True)
        A_eff = A.clone().requires_grad_(True)
        B_eff = B.clone().requires_grad_(True)
        C_eff = C.clone().requires_grad_(True)
        D_eff = D.clone().requires_grad_(True)

        y_eff = selective_scan_efficient(x_eff, delta_eff, A_eff, B_eff, C_eff, D_eff)
        loss_eff = y_eff.sum()
        loss_eff.backward()

        # Compare gradients
        torch.testing.assert_close(
            x_seq.grad, x_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="x gradients should match"
        )
        torch.testing.assert_close(
            delta_seq.grad, delta_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="delta gradients should match"
        )
        torch.testing.assert_close(
            A_seq.grad, A_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="A gradients should match"
        )
        torch.testing.assert_close(
            B_seq.grad, B_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="B gradients should match"
        )
        torch.testing.assert_close(
            C_seq.grad, C_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="C gradients should match"
        )
        torch.testing.assert_close(
            D_seq.grad, D_eff.grad,
            rtol=1e-4, atol=1e-4,
            msg="D gradients should match"
        )

    def test_efficient_gradient_flow(self, small_inputs):
        """Test that gradients flow correctly through efficient scan."""
        x, delta, A, B, C, D = small_inputs

        # Require gradients
        x.requires_grad_(True)
        delta.requires_grad_(True)
        A.requires_grad_(True)
        B.requires_grad_(True)
        C.requires_grad_(True)
        D.requires_grad_(True)

        y = selective_scan_efficient(x, delta, A, B, C, D)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert delta.grad is not None and not torch.all(delta.grad == 0)
        assert A.grad is not None and not torch.all(A.grad == 0)
        assert B.grad is not None and not torch.all(B.grad == 0)
        assert C.grad is not None and not torch.all(C.grad == 0)
        assert D.grad is not None and not torch.all(D.grad == 0)

    def test_efficient_skip_connection_none(self, small_inputs):
        """Test that efficient scan works correctly when D (skip connection) is None."""
        x, delta, A, B, C, _ = small_inputs

        y_seq = selective_scan_seq(x, delta, A, B, C, D=None)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D=None)

        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-5, atol=1e-5,
            msg="Sequential and efficient should match with D=None"
        )

    @pytest.mark.parametrize("ngroups", [1, 2, 4])
    def test_efficient_different_ngroups(self, ngroups):
        """Test efficient scan with different numbers of groups."""
        batch = 2
        seq_len = 64
        d_inner = 128
        d_state = 16

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner)
        delta = torch.randn(batch, seq_len, d_inner)
        A = torch.randn(d_inner, d_state) * 0.5
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-5, atol=1e-5,
            msg=f"Sequential and efficient should match with ngroups={ngroups}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_efficient_cuda(self, small_inputs):
        """Test efficient scan on CUDA if available."""
        x, delta, A, B, C, D = small_inputs

        # Move to CUDA
        x_cuda = x.cuda()
        delta_cuda = delta.cuda()
        A_cuda = A.cuda()
        B_cuda = B.cuda()
        C_cuda = C.cuda()
        D_cuda = D.cuda()

        # Compute on CPU
        y_cpu = selective_scan_efficient(x, delta, A, B, C, D)

        # Compute on CUDA
        y_cuda = selective_scan_efficient(x_cuda, delta_cuda, A_cuda, B_cuda, C_cuda, D_cuda)

        # Compare (move CUDA result back to CPU)
        torch.testing.assert_close(
            y_cpu, y_cuda.cpu(),
            rtol=1e-4, atol=1e-4,
            msg="CPU and CUDA results should match"
        )

    def test_efficient_numerical_stability(self):
        """Test efficient scan numerical stability with extreme values."""
        batch, seq_len, d_inner, d_state, ngroups = 2, 32, 64, 16, 1

        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_inner) * 10  # Large values
        delta = torch.randn(batch, seq_len, d_inner).abs() * 0.1  # Small positive values
        A = torch.randn(d_inner, d_state) * 0.01  # Small A for stability
        B = torch.randn(batch, seq_len, ngroups, d_state)
        C = torch.randn(batch, seq_len, ngroups, d_state)
        D = torch.ones(d_inner)

        y_seq = selective_scan_seq(x, delta, A, B, C, D)
        y_efficient = selective_scan_efficient(x, delta, A, B, C, D)

        # Check no NaN or Inf
        assert not torch.isnan(y_efficient).any(), "Efficient scan produced NaN"
        assert not torch.isinf(y_efficient).any(), "Efficient scan produced Inf"

        # Should still match
        torch.testing.assert_close(
            y_seq, y_efficient,
            rtol=1e-4, atol=1e-4,
            msg="Scans should match even with extreme values"
        )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
