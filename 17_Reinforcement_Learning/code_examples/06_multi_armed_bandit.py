import numpy as np
from typing import List

class Bandit:
    """Represents a single arm in the multi-armed bandit"""
    def __init__(self, mean, std=1.0):
        self.mean = mean
        self.std = std
        self.count = 0
        self.sum = 0
        
    def pull(self):
        """Pull the arm and get reward"""
        return np.random.normal(self.mean, self.std)
    
    def update(self, reward):
        """Update arm statistics"""
        self.count += 1
        self.sum += reward
        
    def estimated_mean(self):
        """Get estimated mean reward"""
        if self.count == 0:
            return 0
        return self.sum / self.count

class EpsilonGreedy:
    """Epsilon-greedy strategy for multi-armed bandit"""
    def __init__(self, bandits: List[Bandit], epsilon=0.1):
        self.bandits = bandits
        self.epsilon = epsilon
        self.total_reward = 0
        self.pulls = 0
        
    def select_arm(self):
        """Select arm using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.bandits))
        else:
            means = [b.estimated_mean() for b in self.bandits]
            return np.argmax(means)
    
    def run(self, num_trials=1000):
        """Run epsilon-greedy algorithm"""
        rewards = []
        for _ in range(num_trials):
            arm = self.select_arm()
            reward = self.bandits[arm].pull()
            self.bandits[arm].update(reward)
            self.total_reward += reward
            self.pulls += 1
            rewards.append(reward)
        return rewards

class ThompsonSampling:
    """Thompson Sampling strategy for multi-armed bandit"""
    def __init__(self, bandits: List[Bandit]):
        self.bandits = bandits
        self.arms = len(bandits)
        self.alphas = np.ones(self.arms)  # Prior successes
        self.betas = np.ones(self.arms)   # Prior failures
        self.total_reward = 0
        self.pulls = 0
        
    def select_arm(self):
        """Select arm using Thompson sampling"""
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)
    
    def run(self, num_trials=1000):
        """Run Thompson sampling"""
        rewards = []
        for _ in range(num_trials):
            arm = self.select_arm()
            reward = self.bandits[arm].pull()
            
            # Update belief (assuming binary reward)
            if reward > np.median(reward):
                self.alphas[arm] += 1
            else:
                self.betas[arm] += 1
                
            self.total_reward += reward
            self.pulls += 1
            rewards.append(reward)
        return rewards

class UCB:
    """Upper Confidence Bound strategy for multi-armed bandit"""
    def __init__(self, bandits: List[Bandit], c=1.0):
        self.bandits = bandits
        self.c = c
        self.counts = np.zeros(len(bandits))
        self.values = np.zeros(len(bandits))
        self.total_reward = 0
        self.pulls = 0
        
    def select_arm(self):
        """Select arm using UCB strategy"""
        ucb_values = self.values + self.c * np.sqrt(np.log(self.pulls + 1) / (self.counts + 1))
        return np.argmax(ucb_values)
    
    def run(self, num_trials=1000):
        """Run UCB algorithm"""
        rewards = []
        for _ in range(num_trials):
            arm = self.select_arm()
            reward = self.bandits[arm].pull()
            
            self.counts[arm] += 1
            self.values[arm] = (self.values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]
            
            self.total_reward += reward
            self.pulls += 1
            rewards.append(reward)
        return rewards

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Create bandits with different means
    bandits = [Bandit(1.0), Bandit(2.0), Bandit(1.5)]
    
    # Epsilon-greedy
    eg = EpsilonGreedy(bandits, epsilon=0.1)
    rewards_eg = eg.run(num_trials=1000)
    print(f"Epsilon-Greedy Average Reward: {eg.total_reward / eg.pulls:.4f}")
    
    # Thompson Sampling
    bandits = [Bandit(1.0), Bandit(2.0), Bandit(1.5)]
    ts = ThompsonSampling(bandits)
    rewards_ts = ts.run(num_trials=1000)
    print(f"Thompson Sampling Average Reward: {ts.total_reward / ts.pulls:.4f}")
    
    # UCB
    bandits = [Bandit(1.0), Bandit(2.0), Bandit(1.5)]
    ucb = UCB(bandits, c=1.0)
    rewards_ucb = ucb.run(num_trials=1000)
    print(f"UCB Average Reward: {ucb.total_reward / ucb.pulls:.4f}")
