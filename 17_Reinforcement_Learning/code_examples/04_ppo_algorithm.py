import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO algorithm"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPONetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = self.fc(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, gae_lambda=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = 0.2  # Clipping parameter
        
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def select_action(self, state):
        """Select action using policy"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits, value = self.network(state_t)
            probs = torch.softmax(logits, dim=1)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0, action])
            
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages(self, next_value):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            td_error = self.rewards[t] + self.gamma * values[t+1] - values[t]
            gae = td_error + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self, next_value, epochs=5):
        """Update policy using PPO"""
        advantages = self.compute_advantages(next_value)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        
        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(self.actions)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        
        for epoch in range(epochs):
            logits, values = self.network(states_t)
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs[range(len(self.actions)), actions_t])
            
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            critic_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
            entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.clear_memory()
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

# Example usage
if __name__ == "__main__":
    env = "CartPole-v1"  # Use your environment
    agent = PPOAgent(state_dim=4, action_dim=2)
    print("PPO Agent initialized")
    print(f"Network: {agent.network}")
