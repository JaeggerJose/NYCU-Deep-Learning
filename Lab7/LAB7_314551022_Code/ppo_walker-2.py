#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from collections import deque
from typing import Deque, List, Tuple
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Neural network layers
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, out_dim)
        self.fc_std = nn.Linear(256, out_dim)
        
        # Initialize weights
        init_layer_uniform(self.fc_mu)
        init_layer_uniform(self.fc_std)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # Create normal distribution
        dist = Normal(mu, std)
        
        # Sample action using reparameterization trick
        action = dist.rsample()
        
        return action, dist

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        # Initialize output layer
        init_layer_uniform(self.fc3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float
) -> List:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    # Compute GAE backwards
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    
    return gae_returns

def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        # Generate a new set of random indices for each epoch
        rand_ids = np.random.permutation(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mini_batch_ids = rand_ids[start:end]
            yield states[mini_batch_ids], actions[mini_batch_ids], values[
                mini_batch_ids
            ], log_probs[mini_batch_ids], returns[mini_batch_ids], advantages[
                mini_batch_ids
            ]

class PPOAgent:
    """PPO Agent."""

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # ADDED: Store initial learning rates for annealing
        self.initial_actor_lr = args.actor_lr
        self.initial_critic_lr = args.critic_lr

        # ADDED: Create result directory
        self.result_dir = "result_task3"
        os.makedirs(self.result_dir, exist_ok=True)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            # FIXED: Detach action from the computation graph before storing
            self.actions.append(selected_action.detach())
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # Calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # Actor loss with clipped objective and entropy
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * dist.entropy().mean()

            # === CHANGED: Critic loss with Value Function Clipping ===
            value = self.critic(state)
            
            # Clip the value function loss
            # 1. Get the predicted value, clipped around the old value
            value_pred_clipped = old_value + torch.clamp(
                value - old_value, -self.epsilon, self.epsilon
            )
            # 2. Calculate the unclipped loss
            value_loss_unclipped = F.mse_loss(value.squeeze(), return_.squeeze())
            # 3. Calculate the clipped loss
            value_loss_clipped = F.mse_loss(value_pred_clipped.squeeze(), return_.squeeze())
            # 4. Take the maximum of the two losses
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
            # ========================================================
            
            # Train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def save_checkpoint(self, episode: int, score: float, average_score: float = None):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_step,
            'score': score,
            'average_score': average_score,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'epsilon': self.epsilon,
                'entropy_weight': self.entropy_weight,
                'update_epoch': self.update_epoch,
                'actor_lr': self.initial_actor_lr,
                'critic_lr': self.initial_critic_lr,
            },
            'env_obs_rms': self.env.obs_rms,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.result_dir, f"ppo_walker_ep{episode}_step{self.total_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if average score is provided and better than previous
        if average_score is not None:
            best_path = os.path.join(self.result_dir, "ppo_walker_best.pt")
            if not os.path.exists(best_path) or average_score > self.best_average_score:
                self.best_average_score = average_score
                torch.save(checkpoint, best_path)
                print(f"New best model saved with average score: {average_score:.2f}")
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        
        # ADDED: Track best average score for model saving
        self.best_average_score = float('-inf')
        
        for ep in tqdm(range(1, self.num_episodes + 1)):
            # ADDED: Learning Rate Annealing
            frac = 1.0 - (ep - 1.0) / self.num_episodes
            new_actor_lr = self.initial_actor_lr * frac
            new_critic_lr = self.initial_critic_lr * frac
            
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = new_actor_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = new_critic_lr

            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                action = action.flatten() 
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    
                    # ADDED: Calculate and display recent 20 episode average
                    recent_scores = scores[-20:] if len(scores) >= 20 else scores
                    recent_avg = np.mean(recent_scores)
                    
                    print(f"Episode {episode_count}: Total Reward = {score:.2f}")
                    
                    wandb.log({
                        "episode_reward": score,
                        "episode": episode_count,
                        "recent_20_avg": recent_avg,
                        "environment_steps": self.total_step
                    })
                    score = 0
                    
                    # ADDED: Status report every 100 episodes
                    if episode_count % 100 == 0:
                        print(f"\n=== STATUS REPORT ===")
                        print(f"Episode: {episode_count}")
                        print(f"Total steps: {self.total_step}")
                        print(f"Recent 20 avg: {recent_avg:.2f}")
                        print(f"Next checkpoint at episode: {((episode_count // 250) + 1) * 250}")
                        print(f"=====================\n")
                    
                    # ADDED: Save model every 250 episodes (moved here to use correct episode_count)
                    if episode_count % 250 == 0 and episode_count > 0:
                        recent_scores = scores[-20:] if len(scores) >= 20 else scores
                        recent_avg = np.mean(recent_scores)
                        print(f"\n=== SAVING CHECKPOINT ===")
                        print(f"Episode: {episode_count}")
                        print(f"Current score: {scores[-1]:.2f}")
                        print(f"Recent 20 avg: {recent_avg:.2f}")
                        self.save_checkpoint(episode_count, scores[-1], recent_avg)
                        print(f"=== CHECKPOINT SAVED ===\n")

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "learning_rate": new_actor_lr,
                "environment_steps": self.total_step
            })
            
            # REMOVED: Save model logic moved above

        self.env.close()

    def test(self, video_folder: str):
        # ... (test function remains unchanged)
        self.is_test = True
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            state = next_state
            score += reward
        print("score: ", score)
        self.env.close()
        self.env = tmp_env
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-full-optimized-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--update-epoch", type=int, default=10)
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    agent.train()