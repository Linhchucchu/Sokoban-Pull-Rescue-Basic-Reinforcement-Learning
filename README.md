<div align="center">

# Reinforcement Learning Course Project
## Topic 15: Single-Box Sokoban with Rescue Pull (1 Use) and Hook Recharge Tile (1 Use)
### Pedagogical Progression based on the Reinforcement Learning Course by David Silver (DeepMind / UCL)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0.0-green.svg?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Core RL](https://img.shields.io/badge/Methodology-100%25%20Tabular%20RL-crimson.svg)](#theoretical-foundations-and-mdp-formulation)
[![Hardware](https://img.shields.io/badge/Execution-Local%20CPU-orange.svg)](#installation-and-execution-pipeline)
[![IEEE Figures](https://img.shields.io/badge/Figures-IEEE%20Format%20(300%20DPI)-darkblue.svg)](#empirical-results-and-visual-gallery)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

<br>

<p align="center">
  <img src="output/videos/map_7x7_dilemma_gameplay.gif" alt="Deadlock Dilemma Gameplay Demo" width="380"/>
  <img src="output/videos/map_7x7_vault_gameplay.gif" alt="Central Vault Gameplay Demo" width="380"/>
</p>

*Figure 1: Autonomous gameplay replays rendered locally. Left: Deadlock Dilemma ($7\times7$, 31 steps, return +69.0). Right: Central Fortified Vault ($7\times7$, 47 steps, return +53.0).*

</div>

---

## 1. Project Overview and Scope

This repository contains the mathematical formulation, modular implementation, and empirical benchmark suite for an extended **Single-Box Sokoban (Topic 15)** environment solved using **Tabular Reinforcement Learning**.

### 1.1 Special Mechanics of Topic 15
* **Rescue Pull (1-time credit)**: When the crate is pushed into a dead corner (Corner Deadlock), the agent can execute an authorized backward pull action to rescue the crate.
* **Hook Recharge Tile (H, 1-time activation)**: Stepping onto the Hook tile recharges the agent's pull capacity back to 1 (holding capacity capped at 1).
* **Intelligent Deadlock Detector**: Automatically distinguishes between irreversible corner deadlocks and salvageable states dynamically based on the active pull credit.

### 1.2 Methodological Constraints
* **100% Core Grading via Tabular RL**: Adheres strictly to the course specification requiring tabular representation for core grading. All solutions are computed explicitly via the **Value Table $V(s)$**, **Q-Table $Q(s, a)$**, and **Policy Table $\pi(s)$**.
* **Pedagogical Progression**: Begins with dynamic programming foundations on the **5x5 basic map** (**Value Iteration, Policy Iteration, Truncated Policy Iteration**) visualized following **David Silver Lecture 3**, before advancing to special mechanics and model-free comparisons.
* **Ground-Truth DP Oracle**: Exact Bellman optimality tables are computed locally as analytical ground truth ($Q^*$) to evaluate true $\text{RMSE}(Q_t, Q^*)$ convergence metrics.
* **Local CPU Execution**: All algorithms run locally on a standard CPU in milliseconds to seconds without requiring external cloud infrastructure.

---

## 2. Benchmark Suite (9 Topological Configurations)

The environment suite spans 9 systematically graded map topologies across $5 \times 5$, $6 \times 6$, and $7 \times 7$ playfields ($|\mathcal{S}|$ ranging from $600$ to $6,240$ states):

<p align="center">
  <img src="figures/maps_all_settings_overview.png" alt="3x3 Master Benchmark Overview" width="850"/>
</p>

*Figure 2: Master Benchmark Suite across 5x5 Open Rooms, 6x6 Intermediate Bridges, and 7x7 Multi-Room Labyrinths.*

### Benchmark Specification Matrix

| Map File | Setting Name | Inner Grid | Free Cells | State Space $|\mathcal{S}|$ | Action Space $|\mathcal{A}|$ | Key Topological Challenge |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `map_5x5_basic.txt` | Setting 1: Basic 5x5 | $5 \times 5$ | 25 | **600** | 4 (No Pull) | Open floor baseline, analytical ground truth ($Q^*$) |
| `map_5x5_inverted.txt` | Variant 5x5: Inverted | $5 \times 5$ | 25 | **600** | 4 (No Pull) | Diagonal symmetry validation (SW to NE) |
| `map_6x6_rescue.txt` | Setting 3: Rescue Pull 6x6 | $6 \times 6$ | 27 | **2,808** | 5 (With Pull) | T-junction corridor, requires 1 rescue pull |
| `map_6x6_labyrinth.txt`| Variant 6x6: Labyrinth | $6 \times 6$ | 27 | **2,808** | 5 (With Pull) | Double-wall maze navigation |
| `map_7x7_deadcorners.txt`| Setting 2: Dead Corners 7x7 | $7 \times 7$ | 37 | **5,328** | 5 (With Pull) | 4 irreversible corner traps |
| `map_7x7_hook.txt` | Setting 4: Hook Recharge 7x7 | $7 \times 7$ | 37 | **5,328** | 5 (With Pull) | L-corridor with Hook recharge shrine |
| `map_7x7_dilemma.txt` | Setting 5: Deadlock Dilemma | $7 \times 7$ | 40 | **6,240** | 5 (With Pull) | Flagship trap: Shortest path does not equal optimal policy |
| `map_7x7_vault.txt` | Variant 7x7: Fortified Vault | $7 \times 7$ | 37 | **5,328** | 5 (With Pull) | Central goal surrounded by fortress walls |
| `map_7x7_zigzag.txt` | Variant 7x7: Zigzag S-Curve | $7 \times 7$ | 37 | **5,328** | 5 (With Pull) | Alternating serpentine horizontal walls |

---

## 3. Theoretical Foundations and MDP Formulation

### 3.1 Formal Markov Decision Process
The environment is modeled as a 5-tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

* **Augmented State Space $\mathcal{S}$**:
  $$s = \big(r_A, c_A, r_B, c_B, \text{pull\_available}, \text{hook\_used}\big) \in \mathcal{S}$$
  The combinatorial state space size is bounded by $|\mathcal{S}| = 4N(N-1)$ for special mode (and $N(N-1)$ for basic mode), where $N$ denotes the number of walkable floor cells. The reachable state space $\mathcal{S}_{\text{reachable}} \subset \mathcal{S}$ represents the exact subset reachable from the initial configuration.
* **Discrete Action Space $\mathcal{A}$**:
  $$\mathcal{A} = \{0: \text{UP},\ 1: \text{DOWN},\ 2: \text{LEFT},\ 3: \text{RIGHT},\ 4: \text{PULL}\}$$
* **Reward Structure $\mathcal{R}(s, a, s')$**:
  $$R(s, a, s') = \begin{cases} 
  +100.0 & \text{if crate reaches Goal pad } G \text{ (terminal)}, \\ 
  -20.0 & \text{if irreversible Deadlock is triggered (terminal)}, \\ 
  -1.0 & \text{step penalty (encourages shortest path)}, \\ 
  -0.5 & \text{rescue pull execution cost.} 
  \end{cases}$$
* **Discount Factor $\gamma = 0.90$**: Enforces Banach contraction mapping while maintaining a long-term planning horizon.

### 3.2 Theorem: Markov Property Preservation
> **Theorem 1 (Markovian Credit Memory)**: In the standard spatial representation $(r_A, c_A, r_B, c_B)$, transitions depend on past action history (whether a rescue pull was already executed or whether the Hook was already consumed), violating the Markov property:
> $$\mathbb{P}(S_{t+1} \mid S_t, A_t, \dots, S_0, A_0) \neq \mathbb{P}(S_{t+1} \mid S_t, A_t)$$
> By augmenting the state vector with binary resource flags $(\text{pull\_available}, \text{hook\_used}) \in \{0, 1\}^2$, all historical dependency is fully encapsulated in the instantaneous state descriptor. Thus:
> $$\mathbb{P}(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} \mid S_t, A_t)$$
> guaranteeing exact validity for Bellman optimality equations and tabular convergence theorems.

### 3.3 Kinematics of Pull Rescue and Hook Recharge
1. **Authorized Pull Condition**: The agent can execute action `PULL` ($a = 4$) if and only if $\text{pull\_available} = 1$, the crate is Manhattan-adjacent to the agent ($\|A - B\|_1 = 1$), and the cell directly behind the agent ($A - (B - A)$) is free of walls and within bounds. Upon execution, the agent steps backward, the crate is towed into the agent's prior location, and $\text{pull\_available}$ is decremented to 0.
2. **Hook Tile Activation**: Stepping onto the Hook tile ($H$) restores $\text{pull\_available}$ from 0 to 1 if and only if $\text{hook\_used} = 0$. If the agent traverses the Hook tile while $\text{pull\_available} = 1$, the Hook is **not** consumed ($\text{hook\_used}$ remains 0).
3. **Deadlock Determination**: A corner configuration is declared an irreversible terminal deadlock if and only if $\text{pull\_available} = 0$ **and** (either no Hook exists, $\text{hook\_used} = 1$, or the agent is path-blocked by walls/crate from reaching $H$). If the Hook is unused and reachable, the episode does not terminate, enabling the agent to execute a planned recharge and rescue.

---

## 4. Empirical Results and Visual Gallery

### 4.1 Stage 1: Dynamic Programming Foundations (David Silver Style)
Solving the Bellman Optimality Equation via **Value Iteration**, **Policy Iteration**, and **Truncated Policy Iteration** on the 5x5 basic map:

<p align="center">
  <img src="figures/david_silver_value_iteration.png" alt="David Silver Value Iteration Progression" width="850"/>
</p>

*Figure 3: David Silver Lecture 3 Visualization: Evolution of Value Function $V_k(s)$ Heatmap and Greedy Policy $\pi_k(s)$ Quiver Grid across iterations $k=0, 1, 2, 3, 10, \text{converged}$.*

<p align="center">
  <img src="figures/dp_algorithms_comparison.png" alt="DP Computational Complexity Comparison" width="480"/>
</p>

*Figure 4: Computational Complexity Comparison: Value Iteration vs Truncated Policy Iteration vs Policy Iteration ($O(n)$ scaling).*

---

### 4.2 Stage 2: 2D Value Slice Analysis (Rescue Pull Mechanism)
Comparing the state-value field $V(s)$ when $\text{pull\_available} = 1$ vs $\text{pull\_available} = 0$:

<p align="center">
  <img src="figures/slice_comparison_pull_vs_nopull.png" alt="2D Value Slice Comparison: Pull=1 vs Pull=0" width="850"/>
</p>

*Figure 5: 2D Value Slice Comparison. Left ($\text{pull}=1$): Dead-corner perimeter cells retain high positive value because errors are reversible via PULL. Right ($\text{pull}=0$): Depletion of pull credit causes the perimeter to collapse into an irreversible terminal deadlock zone (terminal penalty $R_{\text{deadlock}} = -20.0$).*

---

### 4.3 Stage 3: Model-Free Tabular RL (Q-Learning vs SARSA vs Expected SARSA)

<p align="center">
  <img src="figures/td_learning_curves_comparison.png" alt="Head-to-Head TD Learning Curves" width="850"/>
</p>

*Figure 6: Multi-metric Head-to-Head Learning Curves across 20,000 episodes: Cumulative Return, Success Rate, Steps to Goal, and TD Error. SARSA exhibits on-policy risk aversion (steers clear of wall corners); Q-Learning aggressively exploits the optimal boundary path.*

---

### 4.4 Q-Learning 4-Phase Policy Progression ($k = 10, 100, 500, 5,000$)

<p align="center">
  <img src="figures/q_learning_progression_10_100_500_5000.png" alt="Q-Learning 4-Phase Progression" width="850"/>
</p>

*Figure 7: Four pedagogical phases of Tabular Q-Learning: Blind Exploration ($k=10$) to Credit Accumulation ($k=100$) to Policy Breakthrough ($k=500$) to Asymptotic Convergence ($k=5,000$).*

---

### 4.5 Stage 4: Hyperparameter Ablations and 50,000 Episode Scaling

<p align="center">
  <img src="figures/ablation_learning_rate_comparison.png" alt="Learning Rate Ablation" width="420"/>
  <img src="figures/q_learning_1m_episodes_scaling.png" alt="50k Scaling Benchmark" width="420"/>
</p>

*Figure 8: Left: Learning Rate $\alpha \in [0.02, 0.20]$ sensitivity. Right: Large-Scale $50,000$-Episode Scaling on Setting 4 ($|\mathcal{S}| = 5,328$).*

---

### 4.6 Flagship Dilemma Map: Optimal Policy vs Premature Pull Failure
The Deadlock Dilemma ($7\times7$, Setting 5, $|\mathcal{S}| = 6,240$) specifically operationalizes the core challenge of Topic 15: **the pull credit is essential to rescue the crate, but using it prematurely degrades policy performance or precipitates failure**:

| Policy Formulation | Pull Execution Strategy | Solved? | Path Length | Episode Return | Failure Mode / Consequence |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Optimal Policy $\pi^*$** | 2 Pulls (at strategic turnaround) + 1 Hook Recharge | **Yes** | **31 steps** | **$+69.00$** | **Global Optimum**: Navigates around trench to Hook, recharges, and solves. |
| **Premature Early Pull** | Greedy shortcut: pulls crate into left trench at step 2 | **No** | 8 steps | **$-20.00$** | **Deadlock Trap**: Pull credit depleted ($\text{pull}=0$); crate trapped in trench without Hook access. |
| **Myopic Greedy ($\gamma=0.7$)**| Shortest Euclidean heading without long-term planning | **No** | 100 steps | **$-100.00$** | **Timeout Looping**: Future goal value decays ($\gamma^{25} \approx 0.0001$); agent wanders indefinitely. |

---

### 4.7 Reward Formulations & Empirical Optimality Proof
We benchmarked 4 distinct reward shaping configurations on the 6x6 Rescue Map ($|\mathcal{S}| = 2,808$) using Value Iteration and Q-Learning (`experiments/run_reward_ablation.py`):

| Configuration | $R_{\text{step}}$ | $R_{\text{dl}}$ | $R_{\text{pull}}$ | VI Iters | Policy Shift ($\Delta \pi$) | PULL States | Success Rate | Mean Steps | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard Balanced** | **$-1.0$** | **$-20.0$** | **$-0.5$** | **183** | **0 (0.0%)** | **12** | **100.0%** | **21.23** | **Pareto-Optimal Baseline** |
| **Heavy Pull Cost** | $-1.0$ | $-20.0$ | **$-15.0$** | 183 | **170 (6.05%)** | **8** | 99.8% | 21.47 | Rescue-Frugal: switches from PULL to detour push |
| **Severe Deadlock** | $-1.0$ | **$-60.0$** | $-0.5$ | 193 | 0 (0.0%) | 12 | 100.0% | 21.18 | Overly Risk-Averse: slower contraction (+10 iters) |
| **Zero Step Penalty** | **$0.0$** | $-20.0$ | $-0.5$ | 183 | 0 (0.0%) | 12 | **0.4% (TD)** | **99.9** | **Catastrophic Failure**: Lacks temporal urgency; looping timeout |

<p align="center">
  <img src="figures/ablation_reward_configurations.png" alt="Reward Ablation Study" width="850"/>
</p>

*Figure 9: Systematic Reward Formulation Ablation: (a) Rolling Success Rate (demonstrating collapse of Zero Step Penalty), (b) Steps to Goal, and (c) Bellman Error contraction rate.*

---

### 4.8 Multi-Seed Statistical Robustness (5 Independent Seeds)
To guarantee strict reproducibility and satisfy course requirements for variance evaluation, Q-Learning and SARSA were trained across 5 independent random seeds (`[42, 100, 2024, 777, 999]`):

<p align="center">
  <img src="figures/multi_seed_learning_curves.png" alt="Multi-Seed Learning Curves" width="850"/>
</p>

*Figure 10: Multi-seed statistical evaluation: Mean $\pm 1$ Standard Deviation confidence bands across 5 independent random seeds (`experiments/run_multi_seed_evaluation.py`).*

* **Q-Learning (Off-Policy)**: Final Success Rate $= 100.0\% \pm 0.0\%$, Steps to Goal $= 21.2 \pm 0.3$, Return $= +79.6 \pm 0.3$.
* **SARSA (On-Policy)**: Final Success Rate $= 100.0\% \pm 0.0\%$, Steps to Goal $= 21.4 \pm 0.4$, Return $= +79.4 \pm 0.4$.

---

## 5. Interactive Demonstration and Manual Gameplay

An interactive graphical user interface is provided via Pygame for manual playtesting and autonomous policy demonstrations:

### 5.1 Manual Play Mode (Human Testing)
Test game mechanics, collisions, and rescue pull maneuvers manually using keyboard controls:
```bash
# Test the flagship Deadlock Dilemma map (requires 2 pulls and 1 hook recharge):
python demo_pygame.py --manual --map map_7x7_dilemma.txt

# Test other benchmark maps:
python demo_pygame.py --manual --map map_7x7_hook.txt
python demo_pygame.py --manual --map map_7x7_vault.txt
python demo_pygame.py --manual --map map_6x6_rescue.txt
python demo_pygame.py --manual --map map_5x5_basic.txt
```

**Keyboard Controls:**
* **Arrow Keys (Up, Down, Left, Right)**: Move Worker and Push Cargo Crate.
* **SPACE or P**: Execute **Rescue Pull** (tows crate backward, consumes 1 pull credit).
* **R**: Reset episode to initial state.
* **ESC**: Exit window.

### 5.2 Autonomous AI Demonstration
Watch the trained agent execute the optimal policy computed via Tabular RL:
```bash
# Autonomous demonstration on Deadlock Dilemma map:
python demo_pygame.py --map map_7x7_dilemma.txt

# Adjust execution speed (e.g., 3 steps per second for clear inspection):
python demo_pygame.py --map map_7x7_dilemma.txt --fps 3
```

---

## 6. Installation, Execution, and API Usage

### 6.1 Prerequisites and Setup
Recommended Python version: 3.10+
```bash
# Clone the repository
git clone https://github.com/Linhchucchu/Sokoban-Pull-Rescue-Basic-Reinforcement-Learning.git
cd Sokoban-Pull-Rescue-Basic-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Full RL Pipeline Execution
Run the end-to-end experimental workflow locally:

```bash
# Step 1: Dynamic Programming Foundations on 5x5 Map (Computes V*, Q*, pi*)
python run_stage1_basic_dp.py

# Step 2: Special Mechanics Analysis (Pull Rescue, Hook Recharge, Dilemma Map)
python run_stage2_special_mechanics.py

# Step 3: Model-Free Tabular RL (Robbins-Monro, MC, Q-Learning, SARSA on 5x5 & 7x7)
python run_stage3_model_free.py

# Step 4: Systematic Hyperparameter, Reward, and Multi-Seed Ablations
python experiments/run_hyperparameter_ablation.py
python experiments/run_reward_ablation.py
python experiments/run_multi_seed_evaluation.py
python experiments/run_discount_ablation.py
python experiments/run_training_scaling.py --episodes 50000

# Step 5: Export High-Definition Video Suite (MP4 and Animated GIF Replays)
python export_video.py

# Optional: Run verification unit tests
python tests/test_mechanics.py
python tests/test_basic_dp.py
```

### 6.3 Python API Quickstart
The environment complies with the standard Farama Gymnasium interface and can be imported directly into external Python scripts:

```python
from sokoban_env.sokoban_pull_env import SokobanPullEnv

# 1. Initialize environment in special mode (5 actions)
env = SokobanPullEnv("maps/map_7x7_dilemma.txt", mode="special")
obs, info = env.reset()

# 2. Interact via standard Gymnasium step API
# Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: PULL
action = 3  # Move RIGHT
next_obs, reward, terminated, truncated, step_info = env.step(action)

print(f"Reward: {reward}")
print(f"State Tuple: {step_info['state_tuple']}")  # (r_A, c_A, r_B, c_B, pull_avail, hook_used)
print(f"Discrete State Index: {step_info['state_idx']}")
```

### 6.4 Automated Unit Testing
Run the comprehensive unit test suite to verify transition models, Bellman backups, hook deadlock logic, and environment bounds:
```bash
python tests/test_mechanics.py
python tests/test_basic_dp.py
```

### 6.5 Standalone Interactive Notebook
For reviewing the code in an interactive notebook format, a self-contained notebook **[`Sokoban_RL_Master_Colab.ipynb`](Sokoban_RL_Master_Colab.ipynb)** is provided at the repository root.

---

## 7. Repository Structure

```text
sokoban_pull_rescue_rl/
├── sokoban_env/                           # Gymnasium Environment
│   ├── sokoban_pull_env.py                # Core MDP environment (basic and special)
│   └── deadlock_detector.py               # BFS-based Hook-aware corner deadlock detector
│
├── algorithms/                            # 100% Tabular RL Algorithms (Core Scope)
│   ├── dynamic_programming.py             # VI, PI, Truncated PI solvers
│   ├── bellman_evaluation.py              # Matrix Inversion and Iterative PE
│   ├── bellman_optimality.py              # Banach contraction verification
│   ├── stochastic_approximation.py        # Robbins-Monro convergence theorem
│   ├── monte_carlo.py                     # First-visit Incremental MC (earliest-visit update)
│   ├── temporal_difference.py             # Q-Learning, SARSA, Expected SARSA
│   └── bonus_theoretical_extensions/     # Non-graded Deep RL references (DQN, A2C, REINFORCE)
│
├── maps/                                  # 9 Topological Benchmark Maps
│   ├── map_5x5_basic.txt                  # Setting 1: 5x5 Inner Basic (|S| = 600)
│   ├── map_5x5_inverted.txt               # Variant 5x5: Inverted Open (|S| = 600)
│   ├── map_6x6_rescue.txt                 # Setting 3: 6x6 Inner Rescue (|S| = 2,808)
│   ├── map_6x6_labyrinth.txt              # Variant 6x6: Labyrinth Maze (|S| = 2,808)
│   ├── map_7x7_deadcorners.txt            # Setting 2: 7x7 Dead Corners (|S| = 5,328)
│   ├── map_7x7_hook.txt                   # Setting 4: 7x7 Hook Recharge (|S| = 5,328)
│   ├── map_7x7_dilemma.txt                # Setting 5: Deadlock Dilemma (|S| = 6,240)
│   ├── map_7x7_vault.txt                  # Variant 7x7: Fortified Vault (|S| = 5,328)
│   └── map_7x7_zigzag.txt                 # Variant 7x7: Zigzag S-Curve (|S| = 5,328)
│
├── experiments/                           # Ablation Experiments & Statistical Runners
│   ├── run_hyperparameter_ablation.py     # Alpha, epsilon, episodes ablation
│   ├── run_reward_ablation.py             # 4 Reward formulations & policy divergence proof
│   ├── run_multi_seed_evaluation.py       # 5 independent seeds with Mean +/- Std bands
│   ├── run_discount_ablation.py           # Gamma discount factor sensitivity
│   ├── run_training_scaling.py            # Fast scaling runner up to 50k episodes
│   └── run_1m_episodes.py                 # Asymptotic stress-test (1,000,000 episodes)
│
├── visualization/                         # IEEE Visualization and Rendering Suite
│   ├── david_silver_style.py              # Heatmap V(s) + Quiver Policy pi(s)
│   ├── learning_curves.py                 # Multi-metric learning curves with std bands
│   ├── ieee_style.py                      # IEEE double/single column style
│   ├── game_renderer.py                   # High-Definition 3D stone and HUD renderer
│   └── plot_all_maps.py                   # 3x3 Master Overview layout generator
│
├── figures/                               # Publication-ready IEEE Figures (300 DPI)
├── output/                                # Exported Tables (.npy/.csv) and Videos (.mp4/.gif)
├── data/                                  # Oracle ground-truth tables & policy CSVs
├── tests/                                 # Unit test suite
├── demo_pygame.py                         # Interactive Pygame demo (Manual / AI)
├── export_video.py                        # Broadcast-quality video export
├── run_stage1_basic_dp.py                 # Stage 1 DP script
├── run_stage2_special_mechanics.py        # Stage 2 Mechanics script
├── run_stage3_model_free.py               # Stage 3 Model-Free script
├── Sokoban_RL_Master_Colab.ipynb          # Master Interactive Notebook
├── LICENSE                                # MIT License
└── README.md                              # Main documentation
```

> **Strict Grading Compliance Note**: In strict accordance with the Topic 15 guidelines, all core grading algorithms, empirical evaluations, and benchmark results are **100% Tabular Reinforcement Learning**. Deep RL / Function Approximation modules are isolated in `algorithms/bonus_theoretical_extensions/` purely as optional non-graded references.

---

## 8. Quantitative Benchmark Summary (Local CPU Execution)

| Method / Algorithm | Paradigm | Target Environment | Success Rate | Local CPU Wall Time | Convergence Metric |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Matrix Inversion** | Model-Based | 5x5 Basic Map | $100.0\%$ | $1.2\text{ ms}$ | Exact Analytical Ground Truth ($V = (I - \gamma P_\pi)^{-1} R_\pi$) |
| **Value Iteration (VI)** | Model-Based | 5x5 Basic Map | $100.0\%$ | $7.3\text{ ms}$ | Converges in 183 Bellman backups ($\theta = 10^{-7}$) |
| **Policy Iteration (PI)** | Model-Based | 5x5 Basic Map | $100.0\%$ | $0.9\text{ ms}$ | Converges in 7 policy improvement steps |
| **VI on Setting 4 (Hook)** | Model-Based | 7x7 Hook Map | $100.0\%$ | $1.8\text{ s}$ | Solves $|\mathcal{S}| = 5,328$ full state space |
| **VI on Setting 5 (Dilemma)**| Model-Based | 7x7 Dilemma Map | $100.0\%$ | $2.4\text{ s}$ | Optimal path: 31 steps, return $+69.00$ |
| **Monte Carlo (First-visit)**| Model-Free | 5x5 Basic Map | $100.0\%$ | $1.8\text{ s}$ | True earliest-visit episodic returns ($G_t$) |
| **Q-Learning (Off-policy)** | Model-Free Tabular | 5x5 Basic Map | **100.0%** | **$4.0\text{ s}$** ($20k$ eps) | Reaches optimal path, minimal RMSE |
| **SARSA (On-policy)** | Model-Free Tabular | 5x5 Basic Map | **100.0%** | **$3.6\text{ s}$** ($20k$ eps) | Risk-averse, steers clear of corners |
| **Q-Learning on 7x7 Hook** | Model-Free Tabular | 7x7 Hook Map | **100.0%** | **$7.8\text{ s}$** ($10k$ eps) | Reaches goal via Hook recharge |
| **Multi-Seed Variance** | Statistical (5 Seeds)| 5x5 Basic Map | **$100.0\% \pm 0.0\%$** | **$18.5\text{ s}$** | $21.2 \pm 0.3$ steps (Q-Learning) |
| **Q-Learning 50k Scaling** | Model-Free Tabular | 7x7 Hook Map | **100.0%** | **$34.2\text{ s}$** | High state-action coverage ($50.2\%$) |

> **Note on Training Horizons**: Core tabular convergence across all configurations is fully achieved within $5,000$ to $20,000$ episodes. The $50,000$-episode scale benchmark confirms parameter stability without catastrophic forgetting, while `experiments/run_1m_episodes.py` demonstrates asymptotic stress-testing up to $1,000,000$ episodes.

---

## 9. License and Citation

This project is licensed under the MIT License. Developed as an academic implementation for **Course Project Topic 15 (Reinforcement Learning)** based on David Silver's RL Course (DeepMind / UCL).
