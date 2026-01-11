# 3. Q-Learning and Temporal Difference Learning

## Introduction

Temporal Difference (TD) Learning combines ideas from Monte Carlo methods and dynamic programming. Q-Learning is a fundamental model-free algorithm that learns action values directly without knowing the environment model.

## Temporal Difference Learning

### 2.1 Core Idea

TD learning updates value estimates based on other learned estimates (bootstrapping), without waiting for the final outcome.

### 2.2 TD Error

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

The TD error measures the difference between predicted and new estimate.

### 2.3 TD(0) for State Values

```
V(S_t) ← V(S_t) + α × [R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

Where α is the learning rate.

### 2.4 Advantages over Monte Carlo

- Learn before episode ends
- Work with continuing (non-episodic) tasks
- Usually lower variance than MC
- Can use incomplete episodes

## Q-Learning Algorithm

### 3.1 Definition

Q-Learning learns the optimal action-value function Q*(s,a) off-policy:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γmax_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

### 3.2 Off-Policy Learning

- **Behavior Policy**: μ controls exploration (e.g., ε-greedy)
- **Target Policy**: π is what we're learning (greedy w.r.t. Q)
- Q-Learning learns optimal policy while following exploratory policy

### 3.3 Q-Learning Algorithm Steps

```
1. Initialize Q(s, a) arbitrarily for all s, a
2. Repeat for each episode:
   - Initialize state S
   - Repeat for each step:
     a. Choose A from S using policy derived from Q (ε-greedy)
     b. Take action A, observe R, S'
     c. Q(S,A) ← Q(S,A) + α[R + γmax_a Q(S',a) - Q(S,A)]
     d. S ← S'
     e. Until S is terminal
```

### 3.4 Implementation

```python
import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        max_next = np.max(self.Q[next_state]) if not done else 0
        td_target = reward + self.gamma * max_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, env, num_episodes=1000):
        """Train the agent"""
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
```

## SARSA Algorithm

### 4.1 Overview

SARSA (State-Action-Reward-State-Action) is an on-policy TD algorithm:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

### 4.2 Key Difference from Q-Learning

- **SARSA**: Uses actual next action A_{t+1}
- **Q-Learning**: Uses optimal next action
- SARSA is on-policy, Q-Learning is off-policy

### 4.3 Convergence Properties

- SARSA converges to optimal Q values under on-policy conditions
- Q-Learning converges to optimal Q values (off-policy)
- Both guaranteed to converge with table representation

## Expected SARSA

### 5.1 Definition

Expected SARSA averages over next actions:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γE[Q(S_{t+1}, A)] - Q(S_t, A_t)]
```

### 5.2 Advantages

- Lower variance than SARSA
- More stable learning
- Takes into account exploration uncertainty

## Convergence Guarantees

### 6.1 Conditions

For convergence to optimal policy:
1. State-action pairs visited infinitely often
2. Learning rates satisfy: √(α_t) = ∞, ∛(α_t^2) < ∞
3. For Q-Learning: ε-greedy exploration with ε > 0

### 6.2 Convergence Rates

- Q-Learning: O(1/k) convergence
- SARSA: Slower convergence due to exploration
- Both slower than DP but work without model

## Practical Considerations

### 7.1 Function Approximation

For large state spaces, use function approximation:

```python
class DeepQNetwork:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)
    
    def build_model(self, state_size, action_size):
        model = Sequential([
            Dense(64, activation='relu', input_dim=state_size),
            Dense(64, activation='relu'),
            Dense(action_size)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def update(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + gamma * np.max(self.model.predict(next_state))
        self.model.fit(state, target, epochs=1, verbose=0)
```

### 7.2 Experience Replay

Store transitions and sample randomly:
- Breaks correlations
- Improves sample efficiency
- Better learning stability

### 7.3 Target Networks

Use separate networks for stability:
- One network for current Q
- One network (updated periodically) for targets
- Reduces oscillations

## Common Issues and Solutions

1. **Divergence**: Use target networks, smaller learning rates
2. **Instability**: Reduce learning rate, add regularization
3. **Exploration**: Use ε-decay, entropy regularization
4. **Sample Efficiency**: Implement experience replay, prioritized replay

## Summary

Temporal Difference learning and Q-Learning provide powerful model-free methods for RL. TD learning combines the benefits of MC and DP methods, while Q-Learning enables learning of optimal policies even when following exploratory behavior.

## Key Takeaways

- TD learning bootstraps from learned estimates
- Q-Learning is off-policy and learns optimal action values
- SARSA is on-policy and more conservative
- Both converge under appropriate conditions
- Function approximation required for large problems
- Experience replay and target networks improve stability
