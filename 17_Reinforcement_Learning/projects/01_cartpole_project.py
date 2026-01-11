"""
CartPole Control Project

Objective: Train an RL agent to solve the CartPole environment from OpenAI Gym.
The goal is to balance a pole on a moving cart.

Project Steps:
1. Set up the CartPole environment
2. Train a DQN agent
3. Evaluate the trained agent
4. Visualize the results
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CartPoleAgent:
    """DQN Agent for CartPole"""
    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        self.network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor(np.array([x[4] for x in batch]))
        
        target = self.network(states)
        with torch.no_grad():
            next_q_values = self.network(next_states)
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        
        self.optimizer.zero_grad()
        loss = self.criterion(self.network(states), target)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_cartpole(episodes=500):
    """Train the CartPole agent"""
    env = gym.make('CartPole-v1')
    agent = CartPoleAgent()
    
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)
            
            state = next_state
            score += reward
        
        scores.append(score)
        
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode + 1}/{episodes}, Avg Score: {avg_score:.2f}")
    
    env.close()
    return agent, scores

if __name__ == "__main__":
    print("Training CartPole Agent...")
    agent, scores = train_cartpole(episodes=500)
    print(f"Training complete! Final average score: {np.mean(scores[-100:]):.2f}")
