import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import namedtuple

Sample = namedtuple('Sample', ['state', 'action', 'reward', 'next_state', 'done'])

class A3CNetwork(nn.Module):
    """Asynchronous Advantage Actor-Critic network"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(A3CNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = self.fc(state)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

class A3CAgent:
    """Asynchronous Advantage Actor-Critic Agent"""
    def __init__(self, state_dim, action_dim, global_network=None, 
                 lr=1e-4, gamma=0.99, entropy_coeff=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        # Local network
        self.local_network = A3CNetwork(state_dim, action_dim)
        
        # Global network (shared)
        self.global_network = global_network
        if global_network is not None:
            self.local_network.load_state_dict(global_network.state_dict())
        
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)
        self.sample_history = []
        
    def select_action(self, state):
        """Select action using policy"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, value = self.local_network(state_t)
            probs = torch.softmax(policy_logits, dim=1)
            
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action))
            
        return action, log_prob.item(), value.item()
    
    def store_sample(self, state, action, reward, next_state, done):
        """Store transition"""
        self.sample_history.append(Sample(state, action, reward, next_state, done))
    
    def compute_returns(self, next_value):
        """Compute returns and advantages"""
        returns = []
        advantages = []
        
        G = next_value
        for sample in reversed(self.sample_history):
            G = sample.reward + self.gamma * G * (1 - sample.done)
            
            # Get value estimate
            with torch.no_grad():
                state_t = torch.FloatTensor(sample.state).unsqueeze(0)
                _, V = self.local_network(state_t)
                advantage = G - V.item()
            
            returns.insert(0, G)
            advantages.insert(0, advantage)
        
        return returns, advantages
    
    def update(self, returns, advantages):
        """Update local network and sync with global network"""
        states = torch.FloatTensor([s.state for s in self.sample_history])
        actions = torch.LongTensor([s.action for s in self.sample_history])
        returns_t = torch.FloatTensor(returns).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages)
        
        # Forward pass
        policy_logits, values = self.local_network(states)
        probs = torch.softmax(policy_logits, dim=1)
        
        # Policy loss (actor)
        action_log_probs = torch.log(probs[range(len(actions)), actions])
        policy_loss = -(action_log_probs * advantages_t).mean()
        
        # Value loss (critic)
        value_loss = 0.5 * ((values.squeeze() - returns_t.squeeze()) ** 2).mean()
        
        # Entropy regularization
        entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
        
        # Total loss
        total_loss = policy_loss + value_loss - self.entropy_coeff * entropy
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 40.0)
        self.optimizer.step()
        
        # Sync with global network
        if self.global_network is not None:
            self.global_network.load_state_dict(self.local_network.state_dict())
        
        self.sample_history.clear()
        return total_loss.item()

class A3CTrainer:
    """Multi-process A3C trainer"""
    def __init__(self, state_dim, action_dim, num_workers=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_workers = num_workers
        
        self.global_network = A3CNetwork(state_dim, action_dim)
        self.global_network.share_memory()
    
    def worker_fn(self, worker_id, num_episodes=100):
        """Worker process function"""
        agent = A3CAgent(self.state_dim, self.action_dim, 
                        global_network=self.global_network)
        
        for episode in range(num_episodes):
            # Simple dummy environment
            state = np.random.randn(self.state_dim)
            total_reward = 0
            
            for step in range(100):
                action, log_prob, value = agent.select_action(state)
                
                # Dummy reward
                reward = np.random.randn()
                next_state = state + np.random.randn(self.state_dim) * 0.1
                done = np.random.random() < 0.01
                
                agent.store_sample(state, action, reward, next_state, done)
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            # Update after episode
            with torch.no_grad():
                next_value = agent.select_action(state)[2]
            returns, advantages = agent.compute_returns(next_value)
            agent.update(returns, advantages)
            
            if (episode + 1) % 10 == 0:
                print(f"Worker {worker_id}, Episode {episode + 1}, Reward: {total_reward:.2f}")

# Example usage
if __name__ == "__main__":
    trainer = A3CTrainer(state_dim=4, action_dim=2, num_workers=4)
    print("A3C Trainer initialized")
    print(f"Global Network: {trainer.global_network}")
