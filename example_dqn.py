"""
Example DQN Implementation
A complete example showing how to implement Deep Q-Learning for this environment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from game_environment import FireWaterEnv

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network with dueling architecture"""
    
    def __init__(self, state_dim=52, action_dim=6, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages (dueling architecture)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with target network and replay buffer"""
    
    def __init__(
        self,
        state_dim=52,
        action_dim=6,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=100,
        batch_size=64,
        buffer_capacity=100000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        
        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training stats
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current observation
            training: If False, always use greedy policy
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Perform one gradient descent step
        
        Returns:
            loss: TD loss value, or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate actions using target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'updates': self.updates
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        print(f"Model loaded from {path}")


def train_dqn(
    num_episodes=5000,
    max_steps=3000,
    log_frequency=100,
    save_frequency=500,
    use_wandb=False
):
    """Train DQN agents on Fire & Water environment"""
    
    print("=" * 60)
    print("DQN TRAINING")
    print("=" * 60)
    
    # Initialize environment
    env = FireWaterEnv(max_steps=max_steps)
    
    # Initialize agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    fire_agent = DQNAgent(device=device)
    water_agent = DQNAgent(device=device)
    
    # Tracking
    episode_rewards = deque(maxlen=100)
    success_tracker = deque(maxlen=100)
    
    # W&B logging
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="firewater-dqn",
                config={
                    "algorithm": "DQN",
                    "episodes": num_episodes,
                    "max_steps": max_steps
                }
            )
        except ImportError:
            print("Warning: wandb not installed\n")
            use_wandb = False
    
    # Training loop
    for episode in range(num_episodes):
        fire_obs, water_obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select actions
            fire_action = fire_agent.select_action(fire_obs, training=True)
            water_action = water_agent.select_action(water_obs, training=True)
            
            # Step environment
            (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
                fire_action, water_action
            )
            
            # Store experiences
            fire_agent.store_experience(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
            water_agent.store_experience(water_obs, water_action, water_reward, water_next_obs, water_done)
            
            # Update agents
            fire_loss = fire_agent.update()
            water_loss = water_agent.update()
            
            # Track metrics
            episode_reward += (fire_reward + water_reward)
            episode_length += 1
            
            # Update observations
            fire_obs = fire_next_obs
            water_obs = water_next_obs
            done = fire_done or water_done
        
        # Episode complete
        episode_rewards.append(episode_reward)
        success_tracker.append(1 if info['both_won'] else 0)
        
        # Logging
        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(episode_rewards)
            success_rate = np.mean(success_tracker)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Fire Epsilon: {fire_agent.epsilon:.3f}")
            print(f"  Buffer Size: {len(fire_agent.replay_buffer)}")
            
            if use_wandb:
                wandb.log({
                    "episode": episode + 1,
                    "avg_reward": avg_reward,
                    "success_rate": success_rate,
                    "epsilon": fire_agent.epsilon,
                    "buffer_size": len(fire_agent.replay_buffer)
                })
        
        # Save checkpoints
        if (episode + 1) % save_frequency == 0:
            fire_agent.save(f"fire_dqn_ep{episode+1}.pth")
            water_agent.save(f"water_dqn_ep{episode+1}.pth")
    
    # Final save
    fire_agent.save("fire_dqn_final.pth")
    water_agent.save("water_dqn_final.pth")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final success rate: {np.mean(success_tracker):.2%}")
    
    if use_wandb:
        wandb.finish()


def test_dqn(fire_model_path, water_model_path, num_episodes=10):
    """Test trained DQN agents"""
    
    print("Testing trained agents...")
    
    env = FireWaterEnv()
    fire_agent = DQNAgent()
    water_agent = DQNAgent()
    
    # Load trained models
    fire_agent.load(fire_model_path)
    water_agent.load(water_model_path)
    
    successes = 0
    
    for episode in range(num_episodes):
        fire_obs, water_obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use greedy policy (no exploration)
            fire_action = fire_agent.select_action(fire_obs, training=False)
            water_action = water_agent.select_action(water_obs, training=False)
            
            (fire_obs, water_obs), rewards, dones, info = env.step(fire_action, water_action)
            episode_reward += sum(rewards)
            done = dones[0] or dones[1]
        
        if info['both_won']:
            successes += 1
            print(f"Episode {episode + 1}: SUCCESS (reward: {episode_reward:.2f})")
        else:
            print(f"Episode {episode + 1}: Failed (reward: {episode_reward:.2f})")
    
    print(f"\nTest Results: {successes}/{num_episodes} successes ({successes/num_episodes:.1%})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        if len(sys.argv) < 4:
            print("Usage: python example_dqn.py test <fire_model.pth> <water_model.pth>")
        else:
            test_dqn(sys.argv[2], sys.argv[3])
    else:
        # Training mode
        train_dqn(
            num_episodes=5000,
            log_frequency=100,
            save_frequency=500,
            use_wandb=False
        )
