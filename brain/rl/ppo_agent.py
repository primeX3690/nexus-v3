"""
brain/rl/ppo_agent.py

Proximal Policy Optimization with a hand-rolled 2-layer MLP (tanh
hidden layer) for both the discrete policy head and the value head,
including manual backpropagation and an Adam optimizer - all in plain
numpy, because no autodiff framework (torch/jax) is installed in this
CPU-only environment.

Every gradient here (softmax/log-prob gradient, entropy gradient, PPO
clipped-surrogate gradient, tanh backprop) is verified against
numerical finite differences in tests/test_ppo.py - hand-derived
backprop is exactly the kind of code that "looks right" while being
subtly wrong, so it does not ship without that check.
"""
from __future__ import annotations

import numpy as np

from utils.config import PPOConfig
from utils.seeding import get_rng
from utils.math_ops import softmax, clip_grad_norm
from brain.rl.replay_buffer import RolloutBuffer


class MLP:
    """Single-hidden-layer MLP: x -> tanh(x@W1+b1) -> (that)@W2+b2.
    Stores the last forward pass's intermediates so `backward` can be
    called immediately after `forward` (standard for a manual-backprop
    training loop, not meant for arbitrary reuse across calls)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rng: np.random.Generator):
        limit1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        limit2 = np.sqrt(6.0 / (hidden_dim + out_dim))
        self.W1 = rng.uniform(-limit1, limit1, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.uniform(-limit2, limit2, (hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in_dim) -> out: (batch, out_dim)"""
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        self._cache = (x, z1, h1)
        return z2

    def backward(self, d_out: np.ndarray) -> dict[str, np.ndarray]:
        """d_out: (batch, out_dim) = dL/d(z2). Returns param grads
        (already summed over batch, matching how the caller averages
        the loss) and does NOT update params - caller applies the
        optimizer step."""
        x, z1, h1 = self._cache
        batch = x.shape[0]

        dW2 = h1.T @ d_out
        db2 = d_out.sum(axis=0)
        dh1 = d_out @ self.W2.T
        dz1 = dh1 * (1 - np.tanh(z1) ** 2)  # tanh'(z) = 1 - tanh(z)^2
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def apply_grads(self, grads: dict[str, np.ndarray]) -> None:
        self.W1 -= grads["W1"]
        self.b1 -= grads["b1"]
        self.W2 -= grads["W2"]
        self.b2 -= grads["b2"]


class Adam:
    """Standard Adam optimizer over a dict of named param arrays,
    returning per-step UPDATE deltas (not applying them) so it composes
    cleanly with MLP.apply_grads (which does `param -= delta`)."""

    def __init__(self, param_shapes: dict[str, tuple], lr: float = 3e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros(shape) for k, shape in param_shapes.items()}
        self.v = {k: np.zeros(shape) for k, shape in param_shapes.items()}
        self.t = 0

    def compute_update(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.t += 1
        updates = {}
        for k, g in grads.items():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            updates[k] = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updates


def categorical_log_prob(logits: np.ndarray, action: int) -> tuple[float, np.ndarray]:
    """logits: (n_actions,). Returns (log_prob_of_action, full_probs)."""
    probs = softmax(logits)
    return float(np.log(probs[action] + 1e-12)), probs


def entropy_and_grad(probs: np.ndarray) -> tuple[float, np.ndarray]:
    """H(p) and dH/dz (gradient w.r.t. PRE-softmax logits z), derived as:
    dH/dz_j = -p_j * (H + log p_j)."""
    log_p = np.log(probs + 1e-12)
    H = float(-np.sum(probs * log_p))
    dH_dz = -probs * (H + log_p)
    return H, dH_dz


class PPOAgent:
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64,
                 config: PPOConfig | None = None, seed_stream: str = "ppo_agent"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config or PPOConfig()
        rng = get_rng(seed_stream)

        self.policy_net = MLP(state_dim, hidden_dim, n_actions, rng)
        self.value_net = MLP(state_dim, hidden_dim, 1, rng)

        self.policy_opt = Adam({k: v.shape for k, v in self.policy_net.params().items()}, lr=self.config.lr)
        self.value_opt = Adam({k: v.shape for k, v in self.value_net.params().items()}, lr=self.config.lr)

        self.action_rng = get_rng(f"{seed_stream}_actions")

    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        """Returns (action, log_prob, value) for a single state."""
        logits = self.policy_net.forward(state[None, :])[0]
        probs = softmax(logits)
        action = int(self.action_rng.choice(self.n_actions, p=probs))
        log_prob, _ = categorical_log_prob(logits, action)
        value = float(self.value_net.forward(state[None, :])[0, 0])
        return action, log_prob, value

    def _policy_loss_and_grad(self, states, actions, old_log_probs, advantages):
        """All args are (batch, ...) arrays. Returns (scalar_loss, d_logits (batch, n_actions))."""
        logits = self.policy_net.forward(states)  # (B, A)
        probs = softmax(logits, axis=-1)
        B = states.shape[0]

        new_log_probs = np.log(probs[np.arange(B), actions] + 1e-12)
        ratio = np.exp(new_log_probs - old_log_probs)

        clip_eps = self.config.clip_ratio
        surrogate1 = ratio * advantages
        clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        surrogate2 = clipped_ratio * advantages
        policy_loss_per_sample = -np.minimum(surrogate1, surrogate2)

        # d(loss)/d(logits): only flows through the branch the min() selected,
        # and only through `ratio` (the clipped branch is treated as a
        # constant w.r.t. logits once it has saturated - standard PPO
        # gradient behavior, matching torch's clip()).
        use_unclipped = surrogate1 <= surrogate2  # (B,) bool
        onehot = np.zeros_like(probs)
        onehot[np.arange(B), actions] = 1.0
        d_logprob_d_logits = onehot - probs  # (B, A), standard softmax-logprob grad

        d_ratio_d_logits = ratio[:, None] * d_logprob_d_logits  # (B, A)
        d_surrogate1_d_logits = advantages[:, None] * d_ratio_d_logits
        grad_selector = use_unclipped[:, None].astype(np.float64)  # 1 where unclipped branch active
        d_policy_loss_d_logits = -grad_selector * d_surrogate1_d_logits  # 0 where clipped branch won

        # entropy bonus (maximize entropy -> subtract from loss)
        entropies = np.zeros(B)
        d_entropy_d_logits = np.zeros_like(probs)
        for i in range(B):
            H, dH = entropy_and_grad(probs[i])
            entropies[i] = H
            d_entropy_d_logits[i] = dH

        total_loss_per_sample = policy_loss_per_sample - self.config.entropy_coef * entropies
        d_total_d_logits = d_policy_loss_d_logits - self.config.entropy_coef * d_entropy_d_logits

        return float(np.mean(total_loss_per_sample)), d_total_d_logits / B

    def _value_loss_and_grad(self, states, returns):
        values = self.value_net.forward(states)[:, 0]  # (B,)
        diff = values - returns
        loss = float(np.mean(diff ** 2))
        d_loss_d_value = (2.0 * diff / len(returns))[:, None]  # (B,1)
        return loss, d_loss_d_value

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        advantages, returns = buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)
        # Normalize advantages - standard PPO trick, stabilizes the clip
        # region across batches of very different reward scales.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = np.array(buffer.states)
        actions = np.array(buffer.actions)
        old_log_probs = np.array(buffer.log_probs)

        rng = get_rng("ppo_update_shuffle")
        stats = {"policy_loss": [], "value_loss": []}

        for _ in range(self.config.epochs_per_update):
            for batch_idx in buffer.get_batches(self.config.minibatch_size, rng):
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                p_loss, d_logits = self._policy_loss_and_grad(b_states, b_actions, b_old_log_probs, b_advantages)
                policy_grads = self.policy_net.backward(d_logits)
                policy_grads = clip_grad_norm(policy_grads, self.config.max_grad_norm)
                self.policy_net.apply_grads(self.policy_opt.compute_update(policy_grads))

                v_loss, d_value = self._value_loss_and_grad(b_states, b_returns)
                value_grads = self.value_net.backward(d_value)
                value_grads = clip_grad_norm(value_grads, self.config.max_grad_norm)
                self.value_net.apply_grads(self.value_opt.compute_update({
                    k: v * self.config.value_coef for k, v in value_grads.items()
                }))

                stats["policy_loss"].append(p_loss)
                stats["value_loss"].append(v_loss)

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def save(self, path: str) -> None:
        """Save policy+value network weights and optimizer state to a
        single .npz file. Without this, a trained agent's weights vanish
        the moment the process exits - there'd be no way to train once
        and demo later, which is exactly the workflow a fellowship demo
        needs."""
        np.savez(
            path,
            policy_W1=self.policy_net.W1, policy_b1=self.policy_net.b1,
            policy_W2=self.policy_net.W2, policy_b2=self.policy_net.b2,
            value_W1=self.value_net.W1, value_b1=self.value_net.b1,
            value_W2=self.value_net.W2, value_b2=self.value_net.b2,
            policy_opt_t=self.policy_opt.t, value_opt_t=self.value_opt.t,
            **{f"policy_opt_m_{k}": v for k, v in self.policy_opt.m.items()},
            **{f"policy_opt_v_{k}": v for k, v in self.policy_opt.v.items()},
            **{f"value_opt_m_{k}": v for k, v in self.value_opt.m.items()},
            **{f"value_opt_v_{k}": v for k, v in self.value_opt.v.items()},
            state_dim=self.state_dim, n_actions=self.n_actions,
        )

    def load(self, path: str) -> None:
        """Restore weights AND optimizer moment estimates (m, v, t) -
        loading only the weights and resetting Adam's state would cause
        a large effective-learning-rate spike on the first post-load
        update, since Adam's bias-correction assumes t starts small."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        assert int(data["state_dim"]) == self.state_dim, "checkpoint state_dim mismatch"
        assert int(data["n_actions"]) == self.n_actions, "checkpoint n_actions mismatch"

        self.policy_net.W1 = data["policy_W1"]; self.policy_net.b1 = data["policy_b1"]
        self.policy_net.W2 = data["policy_W2"]; self.policy_net.b2 = data["policy_b2"]
        self.value_net.W1 = data["value_W1"]; self.value_net.b1 = data["value_b1"]
        self.value_net.W2 = data["value_W2"]; self.value_net.b2 = data["value_b2"]

        self.policy_opt.t = int(data["policy_opt_t"])
        self.value_opt.t = int(data["value_opt_t"])
        for k in self.policy_opt.m:
            self.policy_opt.m[k] = data[f"policy_opt_m_{k}"]
            self.policy_opt.v[k] = data[f"policy_opt_v_{k}"]
        for k in self.value_opt.m:
            self.value_opt.m[k] = data[f"value_opt_m_{k}"]
            self.value_opt.v[k] = data[f"value_opt_v_{k}"]