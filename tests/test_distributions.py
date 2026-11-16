import math

import torch
from torch import distributions as D

from ppo.distributions import Categorical, DiagGaussian


def test_categorical_log_prob_and_entropy_match_torch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(32, 5) * 3.0
    actions = torch.randint(0, 5, (32,))
    ours, ref = Categorical(logits), D.Categorical(logits=logits)
    torch.testing.assert_close(ours.log_prob(actions), ref.log_prob(actions))
    torch.testing.assert_close(ours.entropy(), ref.entropy())
    torch.testing.assert_close(ours.probs, ref.probs)
    assert torch.equal(ours.mode(), logits.argmax(-1))


def test_categorical_gumbel_max_sampling_frequencies() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0])
    n = 200_000
    samples = Categorical(logits.expand(n, 4)).sample()
    freq = torch.bincount(samples, minlength=4).float() / n
    expected = torch.softmax(logits, -1)
    # 5 standard errors of a binomial proportion.
    tol = 5 * torch.sqrt(expected * (1 - expected) / n)
    assert torch.all((freq - expected).abs() < tol)


def test_categorical_log_prob_gradient_matches_torch() -> None:
    logits = torch.randn(8, 3, requires_grad=True)
    actions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    (g_ours,) = torch.autograd.grad(Categorical(logits).log_prob(actions).sum(), logits)
    (g_ref,) = torch.autograd.grad(D.Categorical(logits=logits).log_prob(actions).sum(), logits)
    torch.testing.assert_close(g_ours, g_ref)


def test_diag_gaussian_log_prob_and_entropy_match_torch() -> None:
    torch.manual_seed(0)
    mean = torch.randn(16, 6)
    log_std = torch.randn(6) * 0.5
    actions = torch.randn(16, 6) * 2.0
    ours = DiagGaussian(mean, log_std)
    ref = D.Independent(D.Normal(mean, log_std.exp().expand_as(mean)), 1)
    torch.testing.assert_close(ours.log_prob(actions), ref.log_prob(actions))
    torch.testing.assert_close(ours.entropy(), ref.entropy())
    # Equivalent to a full multivariate normal with diagonal covariance.
    mvn = D.MultivariateNormal(mean, covariance_matrix=torch.diag_embed(log_std.exp().pow(2).expand_as(mean)))
    torch.testing.assert_close(ours.log_prob(actions), mvn.log_prob(actions), rtol=1e-4, atol=1e-4)


def test_diag_gaussian_sampling_moments() -> None:
    torch.manual_seed(0)
    mean = torch.tensor([1.0, -2.0, 0.5])
    log_std = torch.tensor([0.0, math.log(0.1), math.log(3.0)])
    x = DiagGaussian(mean.expand(100_000, 3), log_std).sample()
    torch.testing.assert_close(x.mean(0), mean, atol=0.03, rtol=0)
    torch.testing.assert_close(x.std(0), log_std.exp(), atol=0.03, rtol=0.01)


def test_diag_gaussian_gradients_flow_to_log_std() -> None:
    mean = torch.zeros(4, 2, requires_grad=True)
    log_std = torch.zeros(2, requires_grad=True)
    DiagGaussian(mean, log_std).entropy().sum().backward()
    # dH/dlog_std_d = batch size (entropy summed over 4 rows).
    torch.testing.assert_close(log_std.grad, torch.full((2,), 4.0))
