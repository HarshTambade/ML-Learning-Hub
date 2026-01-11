"""
Basic Q-Learning Exercise

Objective: Implement a simple Q-learning agent to navigate a gridworld
and collect rewards.

TODO:
1. Initialize the Q-table with random values
2. Implement the select_action function using epsilon-greedy strategy
3. Implement the update_q_table function
4. Train the agent and plot the results
"""

import numpy as np
import matplotlib.pyplot as plt

class GridworldEnvironment:
    """Simple 5x5 gridworld environment"""
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size - 1, grid_size - 1]
        
    def reset(self):
        self.agent_pos = [0, 0]
        return self.get_state()
    
    def get_state(self):
        """Convert position to state number"""
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def step(self, action):
        """Execute action and return new state and reward"""
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return self.get_state(), reward, done

class QLearningAgent:
    """TODO: Implement Q-learning agent"""
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # TODO: Initialize Q-table
        # self.q_table = ?
    
    def select_action(self, state):
        """TODO: Implement epsilon-greedy action selection"""
        # Hint: Use epsilon for exploration vs exploitation
        pass
    
    def update_q_table(self, state, action, reward, next_state, done):
        """TODO: Implement Q-value update"""
        # Hint: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        pass
    
    def train(self, env, num_episodes=100):
        """TODO: Train the agent"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # TODO: Complete the training loop
                # 1. Select action
                # 2. Execute action in environment
                # 3. Update Q-table
                # 4. Update state
                pass
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
        return episode_rewards

# Main execution
if __name__ == "__main__":
    # Initialize environment and agent
    env = GridworldEnvironment(grid_size=5)
    agent = QLearningAgent(state_space_size=25, action_space_size=4)
    
    # Train agent
    rewards = agent.train(env, num_episodes=100)
    
    # Plot results
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Q-Learning Training Progress')
    plt.show()
