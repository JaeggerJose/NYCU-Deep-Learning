# 改進的超參數設定和程式碼修正

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.last_obs = None  # for max-over-two-frames

    def _max_two_frames(self, obs):
        if self.last_obs is None:
            return obs
        return np.maximum(obs, self.last_obs)

    def preprocess(self, obs):
        obs_max = self._max_two_frames(obs)
        gray = cv2.cvtColor(obs_max, cv2.COLOR_RGB2GRAY)
        # 可選：先 crop 再 resize（例如 gray[34:194, :]）
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        self.last_obs = obs
        return resized.astype(np.uint8)

    def reset(self, obs):
        self.last_obs = None
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.episode_losses = []
        input_shape = self.preprocessor.reset(self.env.reset()[0]).shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(input_shape, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        # 重要修正：Pong 的初始 best_reward 應該設為 -21
        self.best_reward = -21  
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.memory = deque(maxlen=args.memory_size)
        
        # 新增：用於追蹤最近的獎勵
        self.recent_rewards = deque(maxlen=100)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().div_(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=20000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 重要：Reward clipping for Atari
                clipped_reward = np.clip(reward, -1, 1)
                
                next_state = self.preprocessor.step(next_obs)
                # 存儲 clipped reward 到 replay buffer
                self.memory.append((state, action, clipped_reward, next_state, done))

                # 在步進時：
                if self.env_count % 4 == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward  # 記錄原始 reward 用於評估
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    avg_recent_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} "
                          f"UC: {self.train_count} Eps: {self.epsilon:.4f} "
                          f"Avg100: {avg_recent_reward:.2f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Avg100_Reward": avg_recent_reward
                    })

            # 追蹤最近的獎勵
            self.recent_rewards.append(total_reward)
            avg_recent_reward = np.mean(self.recent_rewards)
            
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} "
                  f"Avg100: {avg_recent_reward:.2f} SC: {self.env_count} "
                  f"UC: {self.train_count} Eps: {self.epsilon:.4f}")
            
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Avg100_Reward": avg_recent_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })

            if self.episode_losses:
                avg_loss = sum(self.episode_losses) / len(self.episode_losses)
                wandb.log({
                    "Episode": ep,
                    "Avg Loss": avg_loss,
                })
                self.episode_losses.clear()

            # 儲存 checkpoint
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            # 評估並儲存最佳模型
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} "
                      f"Best: {self.best_reward:.2f}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                    "Best Reward": self.best_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        # Sample mini-batch
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device).div_(255.0)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device).div_(255.0)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Calculate Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Gradient clipping (重要！)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Logging
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} "
                  f"Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

        self.episode_losses.append(loss.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 針對 Pong 優化的超參數
    parser.add_argument("--save-dir", type=str, default="./pong_results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-dqn-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.00025)  # 提高學習率
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999975)  # 更快的衰減
    parser.add_argument("--epsilon-min", type=float, default=0.01)  # 降低最小 epsilon
    parser.add_argument("--target-update-frequency", type=int, default=10000)  # 更頻繁更新
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    
    args = parser.parse_args()
    
    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 初始化 wandb
    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    
    # 確保使用 Pong 環境
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run(episodes=20000)  # 可能需要更多 episodes
