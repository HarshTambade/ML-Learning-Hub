# 4. Deep Reinforcement Learning and Advanced Topics

## Introduction

Deep Reinforcement Learning (DRL) combines RL with deep neural networks to handle high-dimensional state spaces. This chapter covers key advances and practical considerations for modern RL systems.

## Deep Q-Networks (DQN)

### 2.1 Motivation

Traditional Q-Learning with lookup tables fails for:
- Continuous state spaces
- High-dimensional states (images, etc.)
- Non-stationary environments

DQN uses neural networks to approximate Q-values.

### 2.2 Key Innovations

1. **Experience Replay**
   - Store transitions in replay buffer
   - Sample mini-batches for training
   - Breaks temporal correlations
   - Improves sample efficiency

2. **Target Network**
   - Separate network for computing targets
   - Updated periodically (every C steps)
   - Stabilizes learning
   - Reduces oscillations

3. **Reward Clipping**
   - Normalize rewards to [-1, 1]
   - Improves learning stability
   - Enables hyperparameter sharing across games

### 2.3 DQN Algorithm

```python
import torch
import torch.nn as nn
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.memory = deque(maxlen=10000)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.gamma = 0.99
        self.epsilon = 1.0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random() < self.epsilon:
            return randint(0, 3)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def replay(self, batch_size):
        batch = sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones)
        
        # Compute Q-learning targets
        with torch.no_grad():
            max_q = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * max_q * (1 - dones)
        
        # Compute loss
        q_values = self.q_network(states).gather(1, actions.view(-1, 1))
        loss = nn.MSELoss()(q_values, targets.view(-1, 1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Policy Gradient Methods

### 3.1 REINFORCE Algorithm

Learn policy π_θ directly:

```
∇J(θ) = E[∇ log π_θ(a|s) × G_t]
```

Update rule:
```
θ ← θ + α × ∇ log π_θ(a|s) × G_t
```

### 3.2 Actor-Critic Methods

Combine policy gradient with value function:
- **Actor**: Policy network π(a|s)
- **Critic**: Value network V(s)

Advantages:
- Reduces variance compared to REINFORCE
- Enables continuous action spaces
- More stable convergence

### 3.3 Advantage Actor-Critic (A2C)

```
Advantage = R_t + γV(S_{t+1}) - V(S_t)
Policy loss = -∇ log π(a|s) × Advantage
Value loss = MSE(V(s), Target)
```

## Advanced Policy Optimization

### 4.1 Trust Region Policy Optimization (TRPO)

Constrains policy updates:
- KL divergence between old and new policy ≤ δ
- Ensures stable learning
- Better convergence guarantees

### 4.2 Proximal Policy Optimization (PPO)

Simplifies TRPO with clipped objective:

```
L^CLIP(θ) = E[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]
```

Benefits:
- Easier to implement than TRPO
- More sample efficient
- Empirically very effective

## Model-Based RL

### 5.1 World Models

Learn dynamics model p(s'|s,a):
- Improves sample efficiency
- Enables planning
- Can transfer to new tasks

### 5.2 Planning Methods

- **Monte Carlo Tree Search**: Explore promising branches
- **Cross-Entropy Method**: Iteratively refine action distribution
- **Dyna**: Combine model-based and model-free learning

## Multi-Agent RL

### 6.1 Challenges

- Non-stationary environment (other agents learning)
- Coordination problems
- Credit assignment across agents

### 6.2 QMIX for Cooperative Tasks

- Individual value functions per agent
- Mixing network combines Q-values
- Enables scalable multi-agent learning

## Exploration Strategies

### 7.1 Beyond ε-Greedy

1. **Upper Confidence Bound (UCB)**: Favor uncertain actions
2. **Thompson Sampling**: Sample from posterior
3. **Curiosity-Driven**: Explore unfamiliar states
4. **Intrinsic Motivation**: Maximize information gain

### 7.2 Curiosity-Driven Exploration

```python
class CuriosityModule:
    def __init__(self, state_size, action_size):
        self.forward_model = ForwardModel(state_size, action_size)
        self.inverse_model = InverseModel(state_size, action_size)
    
    def compute_intrinsic_reward(self, state, action, next_state):
        # Prediction error is intrinsic motivation
        predicted_next = self.forward_model(state, action)
        prediction_error = (next_state - predicted_next).pow(2).mean()
        return prediction_error
```

## Transfer Learning in RL

### 8.1 Domain Adaptation

- Train on source domain (simulation)
- Transfer to target domain (reality)
- Requires careful reward shaping

### 8.2 Meta-Learning

- Learn to learn quickly
- Few-shot adaptation to new tasks
- Enables rapid deployment

## Challenges and Solutions

### 9.1 Common Issues

1. **Instability**: Use target networks, reward clipping, gradient clipping
2. **Sample Inefficiency**: Experience replay, prioritized replay
3. **Overestimation Bias**: Double Q-Learning, clipped rewards
4. **Exploration**: Curiosity, entropy regularization
5. **Non-Stationarity**: Curriculum learning, domain randomization

### 9.2 Debugging RL

- Monitor loss curves, reward progress
- Visualize agent behavior
- Test with deterministic policy
- Check gradient norms
- Validate reward signals

## State-of-the-Art Methods

### 10.1 Recent Advances

- **Model-Agnostic Meta-Learning (MAML)**: Fast adaptation
- **Soft Actor-Critic (SAC)**: Entropy regularization for exploration
- **Rainbow**: Combines multiple DQN improvements
- **MuZero**: Learn without knowing rules

## Practical Applications

### 11.1 Games & Robotics

- AlphaGo: Tree search + neural networks
- Atari: DQN and variants
- Robotics: Control and manipulation

### 11.2 Real-World Deployment

- Reward shaping is crucial
- Safety constraints essential
- Sim-to-real transfer challenging
- Human-in-the-loop learning

## Summary

Deep RL enables agents to solve complex problems from high-dimensional observations. Modern algorithms like PPO and SAC provide stable, sample-efficient learning. However, practical deployment requires careful engineering and domain expertise.

## Key Takeaways

- DQN combines Q-Learning with neural networks via experience replay and target networks
- Policy gradient methods directly optimize the policy
- Actor-Critic methods combine policy and value functions
- PPO is simple yet effective for many tasks
- Exploration and stability are key challenges
- Transfer learning and meta-learning enable faster adaptation
