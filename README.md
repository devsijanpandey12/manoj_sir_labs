# Advanced Intelligent Computing & Learning (AICL 407) — Labs

Repository containing lab assignments, implementations, experimental evaluations, and reports for AICL 407.

**Student:** Sijan Pandey  
**Roll No:** 11  
**Program:** B.Tech in Artificial Intelligence (8th Semester)  

---

## Repository Structure

```text
.
├── lab1/
│   ├── Lab1_FrozenLake_QLearning.ipynb   # Main interactive notebook (FrozenLake 8x8)
│   ├── qlearning.py                      # Tabular Q-Learning and SARSA algorithms
│   ├── experiments.py                    # Multi-seed hyperparameter & ablation experiments
│   ├── experiments.pdf                   # Lab 1 Academic Report (PDF)
│   ├── figures_lab1/                     # Publication figures (Q vs SARSA, heatmaps, etc.)
│   ├── build_lab1_notebook.py            # Generator script for Lab 1 notebook
│   └── make_lab1_notebook.py             # Auxiliary builder script
│
└── lab2/
    ├── Lab2_CartPole_DQN.ipynb           # Main interactive notebook (CartPole-v1 DQN)
    ├── dqn.py                            # PyTorch DQN architecture, replay buffer, agent
    ├── experiments_dqn.py                # 5-seed baseline, target network & buffer ablations
    ├── report_dqn.pdf                    # Lab 2 Academic Report (PDF)
    ├── figures_lab2/                     # Publication figures (learning curves, landscape)
    └── make_lab2_notebook.py             # Generator script for Lab 2 notebook
```

---

## Lab Descriptions

### [Lab 1: Tabular Q-Learning & SARSA on FrozenLake (8×8)](./lab1/)
- **Environment:** Gymnasium `FrozenLake-v1` (Map: `8x8`, stochastic `is_slippery=True`)
- **Key Algorithms:** $\epsilon$-Greedy Policy, Off-policy Q-Learning, On-policy SARSA
- **Key Experiments:** 
  - Off-policy Q-Learning vs. On-Policy SARSA comparative convergence
  - Learning rate ($\alpha$) and discount factor ($\gamma$) grid search
  - Exploration decay schedule analysis (linear vs. exponential)
  - Optimal policy extraction and value heatmaps

### [Lab 2: Deep Q-Networks (DQN) on CartPole-v1](./lab2/)
- **Environment:** Gymnasium `CartPole-v1`
- **Key Architecture:** Deep Q-Network with experience replay and periodic target network updates
- **Key Experiments:**
  - 5-seed baseline evaluation achieving the 475 threshold criteria
  - Target network ablation ($C=1$ vs. $C=500$) demonstrating moving-target divergence
  - Replay buffer capacity ablation ($N=10^3, 10^4, 10^5$)
  - Value landscape and decision boundary mapping
