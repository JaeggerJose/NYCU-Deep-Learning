#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, out_dim)
        self.std_layer = nn.Linear(128, out_dim)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu'); nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu'); nn.init.zeros_(self.fc2.bias)
        initialize_uniformly(self.mean_layer)
        initialize_uniformly(self.std_layer)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = F.softplus(self.std_layer(x)) + 1e-6

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action) - torch.log(
            1 - squashed_action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1)

        # scale to env action space [-2, 2]
        action = squashed_action * 2.0
        
        # 計算 entropy
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu'); nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu'); nn.init.zeros_(self.fc2.bias)
        initialize_uniformly(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.value_layer(x).squeeze(-1)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.save_every = args.save_every
        self.checkpoint_dir = args.checkpoint_dir
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action, log_prob, entropy = self.actor(state_t)

        if not self.is_test:
            self.transition = [state_t, log_prob, entropy]  # 現在保存 3 個元素

        return action.detach().cpu().numpy().astype(np.float32)



    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.transition.extend([next_state, reward, terminated, truncated])
        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 注意：現在是 7 個元素
        state, log_prob, entropy, next_state, reward, terminated, truncated = self.transition

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        # 計算 target value
        with torch.no_grad():
            next_value = self.critic(next_state)
            bootstrap_mask = 0.0 if terminated else 1.0
            bootstrap_mask = torch.as_tensor(bootstrap_mask, dtype=torch.float32, device=self.device)
            target = reward + self.gamma * next_value * bootstrap_mask

        # Critic loss
        value = self.critic(state)
        value_loss = F.mse_loss(value, target)

        # 更新 Critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # 計算 advantage
        advantage = (target - value).detach()
        
        # Policy loss (使用正確的 entropy)
        policy_loss = -(log_prob * advantage + self.entropy_weight * entropy)

        # 更新 Actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()


    def save_checkpoint(self, episode: int, step_count: int, score: float) -> None:
        """Save model and optimizer states."""
        checkpoint = {
            "episode": episode,
            "step_count": step_count,
            "score": float(score),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "gamma": self.gamma,
            "entropy_weight": self.entropy_weight,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "seed": self.seed,
        }
        ckpt_path = os.path.join(self.checkpoint_dir, f"a2c_ep{episode}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            while not done:
                # self.env.render()  # Render the environment
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score
                        })  
                    # periodic checkpointing
                    if ep % self.save_every == 0:
                        self.save_checkpoint(ep, step_count, score)

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

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
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    # 288851, 841308, 871135
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = random.randint(0, 1000000)
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    agent.train()