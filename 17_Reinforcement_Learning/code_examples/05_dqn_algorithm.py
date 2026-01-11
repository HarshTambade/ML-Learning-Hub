import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based RL"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self, batch_size=32):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Sync target network with Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# Example usage
if __name__ == "__main__":
    agent = DQNAgent(state_dim=4, action_dim=2)
    print("DQN Agent initialized")
    print(f"Q-Network: {agent.q_network}")
