# CSCN8020 Assignment 1 - Reinforcement Learning Programming

### Student name: Anthony Izevbokun  
### Student ID: 9016626
### Course: CSCN8020 - Reinforcement Learning Programming

## Summary

This assignment covers four reinforcement learning problems: designing a pick-and-place robot as an MDP, performing two manual value-iteration sweeps on a 2x2 Gridworld, running value iteration and in-place value iteration on a 5x5 Gridworld, and applying off-policy Monte Carlo control with importance sampling on the same 5x5 environment..

## Repository structure

```text
CSCN8020_Assignment1/
|
├── README.md
├── requirements.txt
├── .gitignore
├── CSCN8020_Assignment1.ipynb
|
├── src/
|   ├── __init__.py
|   ├── environments.py
|   ├── agents.py
|   ├── policies.py
|   └── utils.py
|
├── images/
|   └── .gitkeep
|
└── logs/
    └── sample_execution.log
```

## How to run the notebook

1. Clone the repository:

```bash
git clone https://github.com/AntUse/CSCN8020_Assignment1.git
cd CSCN8020_Assignment1
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Open Jupyter:

```bash
jupyter notebook CSCN8020_Assignment1.ipynb
```

5. Run the notebook from top to bottom.

## Files required for evaluation

- `CSCN8020_Assignment1.ipynb`: the single notebook containing all four problems.
- `src/`: object-oriented environment, agent, policy, and utility code.
- `logs/sample_execution.log`: sample algorithm execution log.
- `requirements.txt`: dependencies needed to run the notebook.
- `.gitignore`: excludes virtual environments, cache files, checkpoints, and generated runtime files.

## Assumptions and known limitations

- The discount factor used for gridworld tasks is `gamma = 0.9`.
- Rewards are treated as state-based rewards received upon entering the next state.
- Invalid grid actions keep the agent in the same state.
- The 5x5 action list in the assignment repeats `down`; this solution interprets the repeated action as `left` to form the usual four actions: right, down, left, and up.
- Monte Carlo estimates may not exactly match value iteration after a finite number of episodes because Monte Carlo learns from random sampled episodes.


