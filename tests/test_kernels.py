"""Tests for kernels and HSIC."""
import torch
import sys
sys.path.insert(0, '..')

from sarl.kernels import GaussianKernel, HSIC, HSIC_RFF, ConditionalHSIC


def test_gaussian_kernel():
    kernel = GaussianKernel(bandwidth=1.0)
    X = torch.randn(100, 10)
    K = kernel(X)
    assert K.shape == (100, 100)
    assert torch.allclose(K, K.T, atol=1e-6)
    assert torch.allclose(K.diag(), torch.ones(100), atol=1e-6)
    print("✓ Gaussian kernel tests passed")


def test_hsic_independent():
    torch.manual_seed(42)
    X = torch.randn(500, 5)
    Y = torch.randn(500, 5)
    hsic = HSIC()
    val = hsic(X, Y)
    assert val < 0.1, f"HSIC for independent vars should be small, got {val}"
    print("✓ HSIC independence test passed")


def test_hsic_dependent():
    torch.manual_seed(42)
    X = torch.randn(500, 5)
    Y = X + torch.randn(500, 5) * 0.1
    hsic = HSIC()
    val = hsic(X, Y)
    assert val > 0.1, f"HSIC for dependent vars should be large, got {val}"
    print("✓ HSIC dependence test passed")


def test_hsic_rff():
    torch.manual_seed(42)
    X = torch.randn(200, 5)
    Y = X[:, :3] + torch.randn(200, 3) * 0.5
    exact = HSIC()(X, Y)
    rff = HSIC_RFF(5, 3, num_features=2000)(X, Y)
    assert abs(exact - rff) / (exact + 1e-6) < 1.0
    print("✓ RFF HSIC approximation test passed")


def test_conditional_hsic():
    torch.manual_seed(42)
    n = 300
    Z = torch.randn(n, 3)
    X = Z @ torch.randn(3, 5) + torch.randn(n, 5) * 0.1
    Y = Z @ torch.randn(3, 4) + torch.randn(n, 4) * 0.1
    chsic = ConditionalHSIC(ridge_lambda=0.1)
    val = chsic(X, Y, Z)
    assert val < 0.1
    print("✓ Conditional HSIC test passed")


if __name__ == '__main__':
    test_gaussian_kernel()
    test_hsic_independent()
    test_hsic_dependent()
    test_hsic_rff()
    test_conditional_hsic()
    print("\nAll kernel tests passed!")
