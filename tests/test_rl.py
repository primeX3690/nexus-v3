import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from utils.seeding import set_global_seed, get_rng
from utils.config import PPOConfig
from utils.math_ops import softmax
from brain.rl.replay_buffer import RolloutBuffer
from brain.rl.ppo_agent import MLP, PPOAgent


def test_gae_terminal_step_has_no_bootstrap():
    buf = RolloutBuffer()
    for t in range(5):
        buf.add(np.zeros(2), 0, -0.5, 1.0, 0.0, done=(t == 4))
    adv, ret = buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert np.isclose(adv[-1], 1.0)


def test_mlp_backward_matches_finite_differences():
    rng = get_rng("t")
    mlp = MLP(4, 6, 3, rng)
    x = rng.normal(0, 1, (5, 4))
    out = mlp.forward(x)
    d_out = rng.normal(0, 1, out.shape)
    grads = mlp.backward(d_out)

    def loss(W1, b1, W2, b2):
        z1 = x @ W1 + b1
        h1 = np.tanh(z1)
        z2 = h1 @ W2 + b2
        return np.sum(z2 * d_out)

    eps = 1e-5
    param = mlp.W1
    idx = (0, 0)
    orig = param[idx]
    param[idx] = orig + eps
    lp = loss(mlp.W1, mlp.b1, mlp.W2, mlp.b2)
    param[idx] = orig - eps
    lm = loss(mlp.W1, mlp.b1, mlp.W2, mlp.b2)
    param[idx] = orig
    numerical = (lp - lm) / (2 * eps)
    assert np.isclose(numerical, grads["W1"][idx], rtol=1e-3)


def test_ppo_learns_correct_action_on_toy_bandit():
    set_global_seed(0)
    state = np.array([0.5, -0.3, 0.1, 0.8])
    cfg = PPOConfig(lr=1e-2, epochs_per_update=4, minibatch_size=16, entropy_coef=0.01)
    agent = PPOAgent(state_dim=4, n_actions=3, hidden_dim=16, config=cfg)

    for _ in range(25):
        buf = RolloutBuffer()
        for _ in range(64):
            action, log_prob, value = agent.act(state)
            reward = 1.0 if action == 1 else -1.0
            buf.add(state, action, log_prob, reward, value, done=True)
        agent.update(buf, last_value=0.0)

    logits = agent.policy_net.forward(state[None, :])[0]
    probs = softmax(logits)
    assert probs[1] > 0.6


def test_ppo_checkpoint_save_load_roundtrip(tmp_path):
    set_global_seed(0)
    state = np.array([0.5, -0.3, 0.1, 0.8])
    cfg = PPOConfig(lr=1e-2, epochs_per_update=3, minibatch_size=16)
    agent = PPOAgent(state_dim=4, n_actions=3, hidden_dim=16, config=cfg)

    for _ in range(10):
        buf = RolloutBuffer()
        for _ in range(32):
            a, lp, v = agent.act(state)
            r = 1.0 if a == 1 else -1.0
            buf.add(state, a, lp, r, v, done=True)
        agent.update(buf, last_value=0.0)

    probs_before = softmax(agent.policy_net.forward(state[None, :])[0])

    ckpt_path = str(tmp_path / "ckpt")
    agent.save(ckpt_path)

    fresh = PPOAgent(state_dim=4, n_actions=3, hidden_dim=16, config=cfg, seed_stream="different")
    fresh.load(ckpt_path)
    probs_after = softmax(fresh.policy_net.forward(state[None, :])[0])

    assert np.allclose(probs_before, probs_after)
    assert fresh.policy_opt.t == agent.policy_opt.t


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))