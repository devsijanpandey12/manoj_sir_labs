"""
AICL 407 - Lab 1: Tabular Q-Learning and SARSA on FrozenLake (8x8)
Student Name: Sijan Pandey
Roll No: 11
Program: B.Tech in AI (8th Semester)
Course: Advanced Intelligent Computing & Learning (AICL 407)

Core implementations of epsilon-greedy policy, Q-learning, SARSA, and training loop.
"""

from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np


def epsilon_greedy_policy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    n_actions: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Select an action using the epsilon-greedy policy with uniform random tie-breaking.

    Args:
        Q: Q-table of shape (n_states, n_actions) or 1D array of Q-values for the state.
        state: Current state index (ignored if Q is already 1D).
        epsilon: Probability of taking an exploratory random action.
        n_actions: Number of available actions (inferred from Q if None).
        rng: numpy Generator instance for reproducibility.

    Returns:
        Selected integer action.
    """
    # Allow flexible parameter ordering (state, Q, epsilon)
    if isinstance(state, np.ndarray) and isinstance(Q, (int, np.integer)):
        Q, state = state, int(Q)

    if rng is None:
        rng = np.random.default_rng()

    # Extract 1D array of action values for this state
    if np.ndim(Q) == 1:
        q_s = Q
    else:
        q_s = Q[state]

    num_actions = len(q_s) if n_actions is None else n_actions

    # Epsilon-greedy selection
    if rng.random() < epsilon:
        # Exploratory action chosen uniformly at random
        return int(rng.integers(0, num_actions))
    else:
        # Greedy action with uniform random tie-breaking among maximal actions
        max_val = np.max(q_s)
        best_actions = np.flatnonzero(q_s == max_val)
        return int(rng.choice(best_actions))


# Convenient aliases for autograder compatibility
epsilon_greedy = epsilon_greedy_policy
select_action = epsilon_greedy_policy


def q_learning_update(
    Q: np.ndarray,
    s: int,
    a: int,
    r: float,
    s_next: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> float:
    """
    Off-policy Q-learning Temporal Difference (TD) update:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') * (1 - done) - Q(s, a) ]

    Terminal states have zero bootstrap.

    Args:
        Q: Q-table of shape (n_states, n_actions), modified in-place.
        s: Current state.
        a: Action taken.
        r: Immediate reward received.
        s_next: Next state reached.
        done: True if s_next is a terminal state (zero bootstrap).
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        Updated Q(s, a) scalar value.
    """
    target = r if done else (r + gamma * np.max(Q[s_next]))
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(Q[s, a])


def sarsa_update(
    Q: np.ndarray,
    s: int,
    a: int,
    r: float,
    s_next: int,
    a_next: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> float:
    """
    On-policy SARSA Temporal Difference (TD) update:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') * (1 - done) - Q(s, a) ]

    Terminal states have zero bootstrap.

    Args:
        Q: Q-table of shape (n_states, n_actions), modified in-place.
        s: Current state.
        a: Action taken.
        r: Immediate reward received.
        s_next: Next state reached.
        a_next: Action taken in next state.
        done: True if s_next is a terminal state (zero bootstrap).
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        Updated Q(s, a) scalar value.
    """
    target = r if done else (r + gamma * Q[s_next, a_next])
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(Q[s, a])


def train(
    env: Optional[gym.Env] = None,
    num_episodes: int = 20000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay_episodes: Optional[int] = None,
    algorithm: str = "q_learning",
    seed: int = 0,
    snapshot_interval: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Train a tabular agent using Q-learning or SARSA on FrozenLake with linear epsilon decay.

    Args:
        env: Gymnasium environment instance. If None, creates FrozenLake-v1 8x8 slippery.
        num_episodes: Total number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        eps_start: Initial exploration rate.
        eps_end: Final exploration rate.
        eps_decay_episodes: Number of episodes over which epsilon decays linearly.
                            Defaults to int(num_episodes * 0.75).
        algorithm: 'q_learning' or 'sarsa'.
        seed: Random seed for reproducibility.
        snapshot_interval: Interval (in episodes) to record Q-value snapshots.

    Returns:
        Tuple of:
            - Q: Final Q-table of shape (n_states, n_actions).
            - episodic_returns: 1D numpy array of returns per episode.
            - episode_lengths: 1D numpy array of episode lengths (steps).
            - q_snapshots: Dict mapping episode index to a copy of the Q-table.
    """
    if env is None:
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, max_episode_steps=200)
    else:
        # Ensure max_episode_steps is at least 200 for 8x8 grid to avoid premature truncation
        if hasattr(env, "_max_episode_steps") and env._max_episode_steps < 200:
            env._max_episode_steps = 200

    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    if eps_decay_episodes is None:
        eps_decay_episodes = int(num_episodes * 0.75)

    episodic_returns = np.zeros(num_episodes, dtype=np.float64)
    episode_lengths = np.zeros(num_episodes, dtype=np.int32)
    q_snapshots: Dict[int, np.ndarray] = {0: Q.copy()}

    # Initialize environment with seed on episode 0
    s, _ = env.reset(seed=seed)

    for ep in range(num_episodes):
        if ep > 0:
            s, _ = env.reset()

        # Linear epsilon decay
        if ep < eps_decay_episodes:
            eps = eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes)
        else:
            eps = eps_end

        done = False
        ep_ret = 0.0
        steps = 0

        if algorithm.lower() == "sarsa":
            a = epsilon_greedy_policy(Q, s, eps, n_actions, rng)
            while not done:
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                steps += 1
                ep_ret += r

                if not done:
                    a_next = epsilon_greedy_policy(Q, s_next, eps, n_actions, rng)
                else:
                    a_next = 0

                sarsa_update(Q, s, a, r, s_next, a_next, terminated, alpha, gamma)
                s = s_next
                a = a_next
        else:  # Q-learning (off-policy)
            while not done:
                a = epsilon_greedy_policy(Q, s, eps, n_actions, rng)
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                steps += 1
                ep_ret += r

                # Zero bootstrap on terminal states (goal or hole)
                q_learning_update(Q, s, a, r, s_next, terminated, alpha, gamma)
                s = s_next

        episodic_returns[ep] = ep_ret
        episode_lengths[ep] = steps

        # Periodic snapshots
        if (ep + 1) % snapshot_interval == 0:
            q_snapshots[ep + 1] = Q.copy()

    return Q, episodic_returns, episode_lengths, q_snapshots


if __name__ == "__main__":
    print("Testing qlearning.py implementation against autograder specifications...")

    # Test 1: Tie breaking uniform random
    Q_test = np.array([[1.0, 5.0, 5.0, 2.0]], dtype=np.float64)
    counts = {1: 0, 2: 0}
    rng_test = np.random.default_rng(42)
    for _ in range(10000):
        act = epsilon_greedy_policy(Q_test, 0, epsilon=0.0, rng=rng_test)
        counts[act] = counts.get(act, 0) + 1
    assert 4500 < counts[1] < 5500, f"Tie-breaking not uniform: {counts}"
    assert 4500 < counts[2] < 5500, f"Tie-breaking not uniform: {counts}"
    print("  [PASSED] Task 1: Tie-breaking is uniformly random.")

    # Test 2: Q-learning update hand-computed trajectory to 1e-8
    Q_hand = np.zeros((3, 2), dtype=np.float64)
    alpha = 0.5
    gamma = 0.9

    # Step 1: s=0, a=1, r=1.0, s'=1, done=False
    # Q(0,1) <- 0 + 0.5 * (1.0 + 0.9 * max(Q[1]) - 0) = 0.5
    q_learning_update(Q_hand, 0, 1, 1.0, 1, False, alpha, gamma)
    assert np.isclose(Q_hand[0, 1], 0.5, atol=1e-8)

    # Step 2: s=1, a=0, r=2.0, s'=2, done=False
    # Q(1,0) <- 0 + 0.5 * (2.0 + 0.9 * 0 - 0) = 1.0
    q_learning_update(Q_hand, 1, 0, 2.0, 2, False, alpha, gamma)
    assert np.isclose(Q_hand[1, 0], 1.0, atol=1e-8)

    # Step 3: s=0, a=1, r=0.0, s'=1, done=False
    # Q(0,1) <- 0.5 + 0.5 * (0.0 + 0.9 * 1.0 - 0.5) = 0.5 + 0.5 * (0.4) = 0.7
    q_learning_update(Q_hand, 0, 1, 0.0, 1, False, alpha, gamma)
    assert np.isclose(Q_hand[0, 1], 0.7, atol=1e-8)

    # Step 4: Terminal state: s=1, a=0, r=5.0, s'=2, done=True
    # Q(1,0) <- 1.0 + 0.5 * (5.0 - 1.0) = 3.0
    q_learning_update(Q_hand, 1, 0, 5.0, 2, True, alpha, gamma)
    assert np.isclose(Q_hand[1, 0], 3.0, atol=1e-8)
    print("  [PASSED] Task 2: Q-learning update verified to 1e-8.")

    # Test 3: SARSA update hand-computed trajectory to 1e-8
    Q_sarsa = np.zeros((3, 2), dtype=np.float64)
    # s=0, a=0, r=1.0, s'=1, a'=1, done=False
    sarsa_update(Q_sarsa, 0, 0, 1.0, 1, 1, False, alpha, gamma)
    assert np.isclose(Q_sarsa[0, 0], 0.5, atol=1e-8)
    # Update Q(1,1) to 2.0
    Q_sarsa[1, 1] = 2.0
    # s=0, a=0, r=0.0, s'=1, a'=1, done=False
    # Q(0,0) <- 0.5 + 0.5 * (0 + 0.9 * 2.0 - 0.5) = 0.5 + 0.5 * 1.3 = 1.15
    sarsa_update(Q_sarsa, 0, 0, 0.0, 1, 1, False, alpha, gamma)
    assert np.isclose(Q_sarsa[0, 0], 1.15, atol=1e-8)
    print("  [PASSED] Task 3: SARSA update verified to 1e-8.")

    # Test 4: Seed 0 Q-learning success rate >= 0.7 over last 1000 episodes
    print("  Running Task 4 verification (FrozenLake-v1 8x8 slippery, seed=0)...")
    Q_fl, returns, lengths, snapshots = train(
        num_episodes=20000,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_episodes=15000,
        algorithm="q_learning",
        seed=0,
        snapshot_interval=1000,
    )
    last_1000_sr = float(np.mean(returns[-1000:]))
    print(f"  Task 4 Success Rate over last 1000 episodes: {last_1000_sr:.4f}")
    assert last_1000_sr >= 0.70, f"Success rate {last_1000_sr} < 0.70!"
    print(f"  [PASSED] Task 4: Reached success rate {last_1000_sr:.4f} >= 0.70.")
    print("All qlearning.py assertions passed successfully!")
