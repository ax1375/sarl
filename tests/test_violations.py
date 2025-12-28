"""Tests for violation metrics."""
import torch
import sys
sys.path.insert(0, '..')

from sarl.losses import ViolationMetrics, SoftMin, SARLObjective
from sarl.data import generate_anticausal_data, generate_confounded_descendant_data


def test_violations_shape():
    v = ViolationMetrics(use_rff=False)
    phi_x = torch.randn(100, 32)
    y = torch.randint(0, 2, (100,))
    e = torch.randint(0, 3, (100,))
    result = v(phi_x, y, e)
    assert result.v1.shape == () and result.v2.shape == () and result.v3.shape == ()
    print("✓ Violation shapes test passed")


def test_softmin():
    sm = SoftMin(beta=10.0)
    v = torch.tensor([0.5, 0.3, 0.8])
    result = sm(v)
    assert result >= v.min() - 1e-6
    weights = sm.get_weights(v)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert weights[1] > weights[0] and weights[1] > weights[2]
    print("✓ SoftMin test passed")


def test_objective():
    obj = SARLObjective(lambda_inv=1.0, beta=10.0, representation_dim=32, num_envs=3)
    phi_x = torch.randn(90, 32, requires_grad=True)
    y = torch.randint(0, 2, (90,))
    e = torch.cat([torch.full((30,), i) for i in range(3)])
    pred = torch.randn(90, 2, requires_grad=True)
    loss = obj(phi_x, y, e, pred)
    loss.backward()
    assert phi_x.grad is not None
    print("✓ Objective differentiable test passed")


if __name__ == '__main__':
    test_violations_shape()
    test_softmin()
    test_objective()
    print("\nAll violation tests passed!")
