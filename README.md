# M2I-CWO: Multi-agent DRL for IEEE 802.11 CW Optimization

## Overview
Based on my [prior work](https://github.com/itstuyihao/Multi-variable-DRL-single-agent), this repo extends a single-agent framework to a multi-agent system, and the work is presented in [“A comprehensive multi-agent deep reinforcement learning framework with adaptive interaction strategies for contention window optimization in IEEE 802.11 Wireless LANs,”](https://www.sciencedirect.com/science/article/pii/S2405959525000104), where each station is a DRL agent that learns the optimal CW setting. The goal is to keep collision rates low while boosting throughput and maintaining fairness and latency. The environment is simplified (no full 802.11 PHY, no path loss) and focuses on demonstrating the adaptive backoff logic rather than standard-compliant performance.

## Requirements
- Python ≥ 3.8
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) (fluid API) and [PARL](https://github.com/PaddlePaddle/PARL)
- numpy
- Recommended: run on CPU; no GPU requirement

## Repository Structure
```
multi-variable-drl2/
├── README.md
├── main.py                      # dueling DDQN agent + simplified 802.11 backoff simulator
└── data/50000_simtime/10_nodes/ # created at runtime; stores output.txt, loss.txt, thr.txt, latency.txt
```

## Functions
- `main()`: trains a dueling Double DQN to adapt CWmin/CW threshold for 10 stations over 50,000 s of simulated time; logs throughput, loss, and latency every 500 s.
- `new_resolve(new_cwmin, new_cwthreshold)`: executes one contention-resolution step (slot countdown, success/collision handling, CW updates, timing accumulation).
- `Model`, `DDQN`, `Agent`, `ReplayMemory`: PARL/Paddle components implementing the dueling Double DQN policy, target sync, epsilon-greedy exploration, and experience replay.
- `printStats()` / `printLatency()`: summarize collision rate, aggregate throughput, Jain’s fairness index, and average delay at the end of training/evaluation.

## Quick start
From the repo root, create the log directory (default uses 10 stations) and run the simulator:
```bash
mkdir -p data/50000_simtime/10_nodes
python3 main.py
```
Outputs are written under `data/50000_simtime/10_nodes/` as `output.txt` (reward trace), `loss.txt`, `thr.txt` (throughput), and `latency.txt` (per-interval delay).

## Cite
If you use this code or derive results, please cite:
```
@article{2025473,
  author  = {Tu, Yi-Hao and Ma, Yi-Wei},
  journal = {ICT Express},
  title   = {A comprehensive multi-agent deep reinforcement learning framework with adaptive interaction strategies for contention window optimization in IEEE 802.11 Wireless LANs},
  year    = {2025},
  volume  = {11},
  number  = {3},
  pages   = {473-480},
  keywords= {Adaptive interaction reward function, CW optimization, IEEE 802.11 WLANs, M2I-CWO},
  doi     = {https://doi.org/10.1016/j.icte.2025.01.010}
}
```

## Contact
Open an issue or email itstuyihao@gmail.com for questions.
