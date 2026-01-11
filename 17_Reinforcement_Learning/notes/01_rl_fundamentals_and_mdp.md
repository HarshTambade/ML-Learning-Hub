# 1. Reinforcement Learning Fundamentals and Markov Decision Process (MDP)

## Introduction

Reinforcement Learning (RL) is a paradigm where an agent learns to make sequential decisions by interacting with an environment. Unlike supervised learning, the agent doesn't receive explicit labels but instead receives rewards or punishments based on its actions.

## Key Concepts

### 1.1 RL Framework

**Agent**: The learner or decision-maker
**Environment**: The system the agent interacts with
**State (s)**: Current configuration of the environment
**Action (a)**: Decision made by the agent
**Reward (r)**: Feedback signal from the environment
**Policy (π)**: Mapping from states to actions
**Value Function (V)**: Expected cumulative reward from a state
**Q-Function (Q)**: Expected cumulative reward from a state-action pair

### 1.2 The RL Loop

```
1. Agent observes state s
2. Agent selects action a based on policy π
3. Environment transitions to new state s'
4. Environment provides reward r
5. Repeat from step 1
```

## Markov Decision Process (MDP)

### 2.1 Definition

An MDP is formally defined by:
- **S**: Set of states
- **A**: Set of actions
- **P(s'|s,a)**: Transition probability function
- **R(s,a)**: Reward function
- **γ**: Discount factor (0 ≤ γ ≤ 1)

### 2.2 Markov Property

The Markov property states that the future state depends only on the current state and action, not on the history:

```
P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_1, a_1, ..., s_t, a_t)
```

### 2.3 Return and Discount Factor

**Return (G_t)**: Cumulative discounted reward
```
G_t = r_t + γr_{t+1} + γ^2r_{t+2} + ...
```

- γ close to 0: Prioritize immediate rewards (myopic)
- γ close to 1: Prioritize long-term rewards (farsighted)

## Value Functions

### 3.1 State Value Function V(s)

Expected return starting from state s:
```
V(s) = E[G_t | S_t = s]
```

### 3.2 Action Value Function Q(s,a)

Expected return starting from state s, taking action a:
```
Q(s,a) = E[G_t | S_t = s, A_t = a]
```

### 3.3 Bellman Equation

**For State Value:**
```
V(s) = E[R_{t+1} + γV(S_{t+1}) | S_t = s]
```

**For Action Value:**
```
Q(s,a) = E[R_{t+1} + γmax_a' Q(S_{t+1}, a') | S_t = s, A_t = a]
```

## Policies

### 4.1 Policy Definition

A policy π is a mapping from states to actions:
```
π(a|s) = P(A_t = a | S_t = s)
```

### 4.2 Deterministic vs Stochastic

- **Deterministic**: π(s) returns a single action
- **Stochastic**: π(a|s) returns a probability distribution over actions

### 4.3 Policy Evaluation

Compute V^π(s) for a given policy π:
```
V^π(s) = E[G_t | S_t = s, π]
```

### 4.4 Optimal Policy

A policy π* is optimal if:
```
V^{π*}(s) ≥ V^π(s) for all s and π
```

## Problem Types

### 5.1 Episodic Problems

- Agent-environment interaction breaks into episodes
- Terminal state exists
- Example: Game playing, robotics tasks

### 5.2 Continuing Problems

- No terminal state
- Process continues indefinitely
- Requires discount factor γ < 1
- Example: Autonomous driving, trading

## Exploration vs Exploitation

### 6.1 Trade-off

- **Exploitation**: Use known good actions
- **Exploration**: Try new actions to discover better ones

### 6.2 Strategies

- **ε-greedy**: With probability ε, explore; otherwise exploit
- **Softmax**: Use Boltzmann distribution
- **Upper Confidence Bound (UCB)**: Balance exploration with uncertainty
- **Thompson Sampling**: Maintain posterior over Q-values

## Challenges

1. **Credit Assignment**: Determining which actions led to rewards
2. **Delayed Rewards**: Rewards might come much later
3. **Exploration-Exploitation**: How much to explore vs exploit
4. **Non-stationarity**: Environment might change over time
5. **Curse of Dimensionality**: State space can be huge

## Summary

RL provides a framework for learning optimal behavior through interaction with an environment. MDPs provide a mathematical foundation for modeling these problems. Understanding value functions, policies, and the exploration-exploitation trade-off is crucial for developing effective RL agents.

## Key Takeaways

- RL agents learn by interacting with environments and receiving rewards
- MDPs provide a formal framework for RL problems
- Bellman equations are fundamental for computing value functions
- The discount factor balances immediate and long-term rewards
- Optimal policies can be derived from optimal value functions
