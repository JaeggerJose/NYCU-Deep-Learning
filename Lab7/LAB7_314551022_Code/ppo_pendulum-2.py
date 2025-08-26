#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
import json
import random
from collections import deque
from typing import Deque, List, Tuple

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
        log_std_max: int = 2,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remember to initialize the layer weights
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network with 3 hidden layers
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Output layers for mean and log_std
        self.fc_mean = nn.Linear(64, out_dim)
        self.fc_log_std = nn.Linear(64, out_dim)
        
        # Better initialization for Pendulum
        # Use orthogonal initialization for hidden layers
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
        # Initialize output layers with smaller values
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_log_std.weight, gain=0.01)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.zeros_(self.fc_log_std.bias)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        # Pass through hidden layers with ReLU activation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Get mean and log_std
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        
        # Clamp log_std to be within bounds
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # Add small epsilon to prevent numerical issues
        std = torch.exp(log_std)
        
        # Create Normal distribution
        dist = Normal(mean, std)
        
        # Sample action (don't apply tanh here, let the agent handle it)
        action = dist.rsample()  # Use rsample for reparameterization trick
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remember to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_value = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
        # Initialize value head
        nn.init.orthogonal_(self.fc_value.weight, gain=1.0)
        nn.init.zeros_(self.fc_value.bias)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_value(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    # Generalized Advantage Estimation (GAE)
    # GAE combines TD errors with exponentially-weighted averaging
    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    # Iterate backwards through the trajectory
    for step in reversed(range(len(rewards))):
        # Calculate TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        
        # Update GAE: A_t = δ_t + (γ * τ) * A_{t+1}
        # tau (λ in GAE paper) controls bias-variance tradeoff
        gae = delta + gamma * tau * masks[step] * gae
        
        # The return is value + advantage
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
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
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            # Create new tensor copies to avoid graph conflicts
            batch_states = states[rand_ids, :].clone().detach()
            batch_actions = actions[rand_ids].clone().detach()
            batch_values = values[rand_ids].clone().detach()
            batch_log_probs = log_probs[rand_ids].clone().detach()
            batch_returns = returns[rand_ids].clone().detach()
            batch_advantages = advantages[rand_ids].clone().detach()
            
            yield batch_states, batch_actions, batch_values, batch_log_probs, batch_returns, batch_advantages

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

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
        
        # Add obs_dim attribute needed for update_model
        self.obs_dim = env.observation_space.shape[0]
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer - using learning rates from args
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        
        # Learning rate schedulers for better convergence
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, 
            lambda epoch: 1.0 - (epoch / 100000) * 0.5  # Decay to 50% over training
        )
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(
            self.critic_optimizer,
            lambda epoch: 1.0 - (epoch / 100000) * 0.5
        )

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
        raw_action, dist = self.actor(state)
        
        # For Pendulum, scale action to [-2, 2] instead of [-1, 1]
        if self.is_test:
            selected_action = 2.0 * torch.tanh(dist.mean)
        else:
            selected_action = 2.0 * torch.tanh(raw_action)

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(raw_action)  # Store raw action for log_prob calculation
            self.values.append(value)
            self.log_probs.append(dist.log_prob(raw_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Get next value for GAE computation
        with torch.no_grad():
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
        
        # Normalize advantages across the whole batch for stability
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
            # Train critic first (separate from actor)
            self.critic_optimizer.zero_grad()
            value = self.critic(state)
            critic_loss = F.mse_loss(value, return_)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

            # Train actor (completely separate computation)
            self.actor_optimizer.zero_grad()
            
            # Get new action distribution
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # PPO-Clip objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Add entropy bonus
            entropy = dist.entropy().mean()
            actor_loss = policy_loss - self.entropy_weight * entropy
            
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        import os
        
        self.is_test = False
        
        # Create checkpoint directory
        checkpoint_dir = "task2_result"
        os.makedirs(checkpoint_dir, exist_ok=True)

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        recent_scores = deque(maxlen=20)  # Track last 20 episodes for averaging
        score = 0
        episode_count = 0
        best_avg_score = -float('inf')
        rollout_count = 0
        max_steps = 200000  # Maximum environment steps
        target_reached = False
        
        print(f"Training PPO for maximum {max_steps} environment steps...")
        print(f"Target: Average score >= -150 over 20 episodes")
        print(f"Checkpoints will be saved every 25 episodes to '{checkpoint_dir}/'")
        print("="*60)
        
        with tqdm(total=max_steps, desc="Training Progress") as pbar:
            while self.total_step <= max_steps and not target_reached:
                rollout_count += 1
                rollout_score = 0
                
                # Collect rollout
                for _ in range(self.rollout_len):
                    if self.total_step > max_steps:
                        break
                        
                    self.total_step += 1
                    pbar.update(1)
                    
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)

                    state = next_state
                    score += reward[0][0]
                    rollout_score += reward[0][0]

                    # if episode ends
                    if done[0][0]:
                        episode_count += 1
                        state, _ = self.env.reset(seed=self.seed)
                        state = np.expand_dims(state, axis=0)
                        scores.append(score)
                        recent_scores.append(score)
                        
                        # Calculate average score
                        if len(recent_scores) >= 20:
                            avg_score = np.mean(recent_scores)
                        else:
                            avg_score = np.mean(scores) if scores else score
                        
                        # Update progress bar description
                        pbar.set_description(f"Training (Ep {episode_count}, Avg: {avg_score:.1f})")
                        
                        print(f"\nEpisode {episode_count}: Score = {score:.2f}, "
                              f"Avg(last 20) = {avg_score:.2f}, "
                              f"Steps = {self.total_step}")
                        
                        # Log to wandb
                        wandb.log({
                            "episode_reward": score,
                            "average_reward_20": avg_score,
                            "episode": episode_count,
                            "total_steps": self.total_step
                        })
                        
                        # Save checkpoint every 25 episodes
                        if episode_count % 25 == 0:
                            checkpoint_path = os.path.join(
                                checkpoint_dir, 
                                f"ppo_pendulum_ep{episode_count}_step{self.total_step}.pt"
                            )
                            self.save_checkpoint(checkpoint_path, avg_score)
                            print(f"💾 Checkpoint saved: {checkpoint_path}")
                        
                        # Save best model
                        if avg_score > best_avg_score:
                            best_avg_score = avg_score
                            best_model_path = os.path.join(checkpoint_dir, "ppo_pendulum_best.pt")
                            self.save_checkpoint(best_model_path, avg_score)
                            print(f"🏆 New best model saved! Avg Score: {avg_score:.2f}")
                        
                        # Early stopping if target reached
                        if len(recent_scores) >= 20 and avg_score >= -150:
                            print(f"\n🎉 Target score reached! Average score: {avg_score:.2f}")
                            print(f"Total environment steps: {self.total_step}")
                            final_path = os.path.join(
                                checkpoint_dir, 
                                f"ppo_pendulum_success_step{self.total_step}.pt"
                            )
                            self.save_checkpoint(final_path, avg_score)
                        
                        score = 0

                # Update model after rollout
                if len(self.states) > 0 and not target_reached:  # Only update if we have collected data
                    actor_loss, critic_loss = self.update_model(next_state)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    
                    # Log losses to wandb
                    wandb.log({
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "rollout": rollout_count,
                        "total_steps": self.total_step
                    })

        # Save final model
        final_path = os.path.join(checkpoint_dir, f"ppo_pendulum_final_step{self.total_step}.pt")
        final_avg_score = np.mean(recent_scores) if recent_scores else np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores) if scores else -float('inf')
        self.save_checkpoint(final_path, final_avg_score)
        
        print("\n" + "="*60)
        print("Training Summary:")
        print(f"  Total steps: {self.total_step}")
        print(f"  Total episodes: {episode_count}")
        print(f"  Final average score (last 20): {final_avg_score:.2f}")
        print(f"  Best average score achieved: {best_avg_score:.2f}")
        print(f"  Target reached: {'Yes ✅' if target_reached else 'No ❌'}")
        print(f"  Models saved in: {checkpoint_dir}/")
        print("="*60)
        
        # termination
        self.env.close()
    
    def save_checkpoint(self, filepath: str, avg_score: float):
        """Save model checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_step,
            'average_score': avg_score,
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon': self.epsilon,
                'batch_size': self.batch_size,
                'entropy_weight': self.entropy_weight,
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if 'critic_optimizer_state_dict' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
        print(f"Model trained for {checkpoint.get('total_steps', 'unknown')} steps")
        print(f"Average score: {checkpoint.get('average_score', 'unknown'):.2f}")

    def test(self, video_folder: str = None):
        """Test the agent."""
        self.is_test = True
        
        # Setup video recording if folder specified
        if video_folder:
            tmp_env = self.env
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        # Run 20 test episodes
        test_scores = []
        print("\n" + "="*50)
        print("Starting evaluation for 20 episodes...")
        print("="*50)
        
        for episode in range(20):
            state, _ = self.env.reset(seed=episode)  # Different seed for each test episode
            done = False
            score = 0
            step_count = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state[0]  # Remove extra dimension
                score += reward[0][0]
                step_count += 1

            test_scores.append(score)
            print(f"Episode {episode + 1}: Score = {score:.2f}, Steps = {step_count}")

        # Calculate statistics
        avg_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        min_score = np.min(test_scores)
        max_score = np.max(test_scores)
        
        print("\n" + "="*50)
        print("Evaluation Results:")
        print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Min Score: {min_score:.2f}")
        print(f"Max Score: {max_score:.2f}")
        print(f"Success Rate (>-150): {sum(s > -150 for s in test_scores) / 20 * 100:.1f}%")
        print("="*50)
        
        # Log to wandb if available
        try:
            wandb.log({
                "test_average_score": avg_score,
                "test_std_score": std_score,
                "test_min_score": min_score,
                "test_max_score": max_score,
            })
        except:
            pass
        
        # Restore original environment
        if video_folder:
            self.env.close()
            self.env = tmp_env
        
        return avg_score, test_scores
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], 
                        help="Training or testing mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for testing")
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)  # Same as actor for stability
    parser.add_argument("--discount-factor", type=float, default=0.99)  # Higher for long-term rewards
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-weight", type=float, default=0.01)  # Encourage exploration
    parser.add_argument("--tau", type=float, default=0.95)  # GAE lambda
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)  # PPO clip
    parser.add_argument("--rollout-len", type=int, default=2048)  # Standard PPO rollout
    parser.add_argument("--update-epoch", type=int, default=10)  # PPO epochs
    parser.add_argument("--video", action="store_true", help="Record video during testing")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # Create agent
    agent = PPOAgent(env, args)
    
    if args.mode == "train":
        # Initialize wandb for training
        wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True, config=vars(args))
        
        print("="*60)
        print("PPO Training on Pendulum-v1")
        print("="*60)
        print(f"Hyperparameters:")
        print(f"  Actor LR: {args.actor_lr}")
        print(f"  Critic LR: {args.critic_lr}")
        print(f"  Discount Factor: {args.discount_factor}")
        print(f"  GAE Lambda (tau): {args.tau}")
        print(f"  PPO Epsilon: {args.epsilon}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Rollout Length: {args.rollout_len}")
        print(f"  Update Epochs: {args.update_epoch}")
        print(f"  Entropy Weight: {args.entropy_weight}")
        print("="*60)
        
        agent.train()
        
        # Test the final model
        print("\nEvaluating final model...")
        avg_score, _ = agent.test()
        
        wandb.finish()
        
    elif args.mode == "test":
        if args.checkpoint is None:
            print("Error: Please provide a checkpoint path using --checkpoint")
            exit(1)
        
        # Load checkpoint
        agent.load_checkpoint(args.checkpoint)
        
        # Test the model
        video_folder = "test_videos" if args.video else None
        avg_score, test_scores = agent.test(video_folder)
        
        # Save test results
        import json
        results = {
            "checkpoint": args.checkpoint,
            "average_score": float(avg_score),
            "scores": [float(s) for s in test_scores],
            "seed": args.seed
        }
        
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nTest results saved to test_results.json")