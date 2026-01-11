# 2. Value Iteration and Policy Iteration Algorithms

## Introduction

Value Iteration and Policy Iteration are two fundamental dynamic programming algorithms for solving MDPs. They compute optimal value functions and policies by iteratively applying Bellman equations.

## Policy Iteration

### 2.1 Overview

Policy Iteration alternates between:
1. **Policy Evaluation**: Compute V^π(s) for current policy π
2. **Policy Improvement**: Compute a better policy π'

### 2.2 Algorithm Steps

```
1. Initialize π(s) randomly for all states s
2. Repeat until convergence:
   a. Policy Evaluation:
      - Repeat until V converges:
        - For each state s:
          V(s) ← E[R_{t+1} + γV(S_{t+1}) | S_t = s, π]
   b. Policy Improvement:
      - For each state s:
        a* ← argmax_a E[R(s,a) + γV(S_{t+1}) | S_t = s, a]
        π'(s) ← a*
      - If π' = π, return π and V
      - Else, π ← π'
```

### 2.3 Convergence Properties

- **Guaranteed Convergence**: Converges in finite iterations
- **Optimality**: Converges to optimal policy π* and V*
- **Complexity**: Slower per iteration but fewer iterations needed

### 2.4 Implementation Considerations

```python
def policy_iteration(env, gamma, theta=0.0001):
    V = np.zeros(env.nS)
    policy = np.ones((env.nS, env.nA)) / env.nA  # Start with uniform policy
    
    policy_stable = False
    while not policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(env.nS):
                v = V[s]
                V[s] = sum(policy[s, a] * sum(p * (r + gamma * V[s_next]) 
                           for p, s_next, r, _ in env.P[s][a])
                           for a in range(env.nA))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_next, r, _ in env.P[s][a]:
                    action_values[a] += p * (r + gamma * V[s_next])
            new_action = np.argmax(action_values)
            if old_action != new_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[new_action]
    
    return policy, V
```

## Value Iteration

### 3.1 Overview

Value Iteration combines policy evaluation and improvement in a single update:

```
V_{k+1}(s) ← max_a E[R(s,a) + γV_k(S_{t+1}) | S_t = s, a]
```

### 3.2 Algorithm Steps

```
1. Initialize V(s) = 0 for all states
2. Repeat until V converges:
   - For each state s:
     V(s) ← max_a E[R(s,a) + γV(S_{t+1}) | S_t = s, a]
3. Derive policy from final V:
   - π(s) ← argmax_a E[R(s,a) + γV(S_{t+1}) | S_t = s, a]
```

### 3.3 Convergence Properties

- **Guaranteed Convergence**: Converges to V*
- **Optimality**: Resulting policy is optimal
- **Complexity**: Fewer iterations than Policy Iteration, faster convergence

### 3.4 Implementation

```python
def value_iteration(env, gamma, theta=0.0001):
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_next]) 
                          for p, s_next, r, _ in env.P[s][a])
                       for a in range(env.nA))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_next, r, _ in env.P[s][a]:
                action_values[a] += p * (r + gamma * V[s_next])
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0
    
    return policy, V
```

## Comparison

### 4.1 Key Differences

| Aspect | Policy Iteration | Value Iteration |
|--------|-----------------|----------------|
| Approach | Two-step (evaluate, improve) | One-step update |
| Per-iteration cost | Higher (policy eval loop) | Lower |
| Iterations needed | Fewer | More |
| Total computation | Often faster | Often slower |
| Policy stability | Explicit | Implicit |

### 4.2 When to Use

**Policy Iteration**:
- Small state spaces
- When policy is needed frequently
- When computation per iteration is cheap

**Value Iteration**:
- Large state spaces
- When V is primary interest
- When iterations should be fast

## Asynchronous Dynamic Programming

### 5.1 Motivation

Wait-synchronous updates can be inefficient. Asynchronous versions update states individually without sweeping all states.

### 5.2 Variants

- **In-place**: Use updated V values immediately
- **Gauss-Seidel**: Sequential updates within iteration
- **Random updates**: Select states randomly
- **Prioritized sweeping**: Focus on high-priority states

## Computational Complexity

### 6.1 Space Complexity
- V table: O(|S|)
- Policy table: O(|S| × |A|)
- Transition model: O(|S| × |A| × |S|)

### 6.2 Time Complexity
- Policy Iteration: O(n² × m) per cycle
  - n = number of states
  - m = number of actions
- Value Iteration: O(n × m × n) per iteration

## Challenges and Limitations

1. **Large State Spaces**: Infeasible for continuous states
2. **Model Requirements**: Need full transition dynamics
3. **Convergence Speed**: Can be slow for large problems
4. **Memory Requirements**: Must store V and/or policy tables

## Summary

Both Policy Iteration and Value Iteration are guaranteed to find optimal policies, but they differ in computational efficiency. Policy Iteration often converges in fewer iterations but requires expensive policy evaluation steps, while Value Iteration is simpler but may need more iterations.

## Key Takeaways

- Policy Iteration: Explicit two-step approach (evaluate, then improve)
- Value Iteration: Combines evaluation and improvement in one step
- Both guaranteed to converge to optimal solution
- Choice depends on problem characteristics and computational resources
- Asynchronous variants can improve efficiency for large problems
