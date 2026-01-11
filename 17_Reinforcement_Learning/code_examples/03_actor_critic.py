import torch
import torch.nn as nn
import numpy as np

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Sequential(shared, nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(shared, nn.Linear(hidden_size, 1))
    
    def forward(self, state):
        return self.actor(state), self.critic(state)

class ActorCriticAgent:
    """Actor-Critic with Advantage"""
    def __init__(self, state_size, action_size, lr=0.001):
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = 0.99
    
    def train_episode(self, env):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            policy, value = self.network(state_t)
            action = torch.multinomial(policy, 1).item()
            
            next_state, reward, done, _, _ = env.step(action)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.network(next_state_t)
            
            advantage = reward + self.gamma * next_value.item() - value.item()
            
            policy_loss = -torch.log(policy[0, action]) * advantage
            value_loss = advantage ** 2
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            state = next_state
            total_reward += reward
        
        return total_reward
