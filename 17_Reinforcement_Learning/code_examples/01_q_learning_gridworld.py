import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        self.agent_pos = self.start
        self.actions = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up
    
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        x, y = self.agent_pos
        dx, dy = self.actions[action]
        new_x, new_y = x + dx, y + dy
        
        # Boundary check
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.agent_pos = (new_x, new_y)
        
        reward = 1.0 if self.agent_pos == self.goal else -0.01
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, grid_size=5, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.grid_size = grid_size
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        max_next_q = 0 if done else np.max(self.q_table[next_state])
        td_error = reward + self.gamma * max_next_q - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def train(self, env, episodes=500):
        rewards = []
        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done and episode_reward > -10:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        return rewards
    
    def test(self, env, episodes=10):
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
    env = GridWorld(grid_size=5)
    agent = QLearningAgent(grid_size=5)
    rewards = agent.train(env, episodes=500)
    success_rate = agent.test(env, episodes=100)
    print(f"Success Rate: {success_rate*100:.1f}%")
    plt.plot(rewards)
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode')
    plt.show()
