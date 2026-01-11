import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class SARSAAgent:
    """SARSA: On-Policy Temporal Difference Learning"""
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update using next action"""
        if done:
            next_q = 0
        else:
            next_q = self.q_table[next_state][next_action]
        
        td_error = reward + self.gamma * next_q - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def train(self, env, episodes=500):
        """Train SARSA agent"""
        rewards = []
        for ep in range(episodes):
            state = env.reset()
            action = self.get_action(state)
            episode_reward = 0
            done = False
            
            while not done and episode_reward > -10:
                next_state, reward, done = env.step(action)
                next_action = self.get_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                episode_reward += reward
                state = next_state
                action = next_action
            
            rewards.append(episode_reward)
        return rewards
    
    def test(self, env, episodes=10):
        """Test SARSA agent"""
        success_count = 0
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[state])
                state, _, done = env.step(action)
            if state == env.goal:
                success_count += 1
        return success_count / episodes

if __name__ == "__main__":
    from gridworld import GridWorld
    env = GridWorld(grid_size=5)
    agent = SARSAAgent(num_states=25, num_actions=4)
    rewards = agent.train(env, episodes=500)
    success_rate = agent.test(env, episodes=100)
    print(f"SARSA Success Rate: {success_rate*100:.1f}%")
    plt.plot(rewards, label='SARSA')
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()
