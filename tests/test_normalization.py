import numpy as np

from ppo.normalization import ObsNormalizer, RewardScaler, RunningMeanStd


def test_running_mean_std_matches_numpy_over_batches() -> None:
    rng = np.random.default_rng(0)
    batches = [rng.normal(loc=3.0, scale=2.0, size=(rng.integers(1, 50), 4)) for _ in range(30)]
    rms = RunningMeanStd((4,), epsilon=1e-12)
    for b in batches:
        rms.update(b)
    data = np.concatenate(batches)
    np.testing.assert_allclose(rms.mean, data.mean(0), rtol=1e-9)
    np.testing.assert_allclose(rms.var, data.var(0), rtol=1e-6)
    assert np.isclose(rms.count, len(data))


def test_running_mean_std_single_samples() -> None:
    rms = RunningMeanStd((), epsilon=1e-12)
    xs = np.arange(10, dtype=np.float64)
    for x in xs:
        rms.update(np.array([x]))
    assert np.isclose(rms.mean, xs.mean()) and np.isclose(rms.var, xs.var())


def test_obs_normalizer_standardises_and_clips() -> None:
    rng = np.random.default_rng(1)
    norm = ObsNormalizer((3,), clip=5.0)
    data = rng.normal(loc=[10, -5, 0], scale=[1, 10, 0.1], size=(5000, 3))
    out = norm(data)
    np.testing.assert_allclose(out.mean(0), 0.0, atol=0.05)
    np.testing.assert_allclose(out.std(0), 1.0, atol=0.05)
    extreme = norm(np.array([[1e6, 0, 0]]), update=False)
    assert extreme[0, 0] == 5.0
    assert out.dtype == np.float32


def test_obs_normalizer_frozen_does_not_update() -> None:
    norm = ObsNormalizer((2,))
    norm(np.ones((10, 2)))
    state = norm.rms.state_dict()
    norm(np.full((10, 2), 100.0), update=False)
    np.testing.assert_array_equal(norm.rms.mean, state["mean"])


def test_reward_scaler_tracks_discounted_returns_and_resets() -> None:
    gamma = 0.9
    scaler = RewardScaler(num_envs=2, gamma=gamma, clip=1e9)
    scaler(np.array([1.0, 2.0]), np.array([False, True]))
    # Env 1 finished: its accumulator restarts; env 0 keeps discounting.
    np.testing.assert_allclose(scaler.returns, [1.0, 0.0])
    out = scaler(np.array([1.0, 1.0]), np.array([False, False]))
    np.testing.assert_allclose(scaler.returns, [1.0 * gamma + 1.0, 1.0])
    # Scaling divides by the std of returns seen so far and preserves sign.
    assert np.allclose(out, 1.0 / np.sqrt(scaler.rms.var + scaler.epsilon))
    neg = scaler(np.array([-3.0, -3.0]), np.array([False, False]))
    assert np.all(neg < 0)
