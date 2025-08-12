# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

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

#gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("eps_weight", torch.zeros(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_bias", torch.zeros(out_features))

        self.reset_parameters(sigma0)
        self.reset_noise()

    def reset_parameters(self, sigma0):
        mu_range = 1 / np.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)

        self.sigma_weight.data.fill_(sigma0 / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(sigma0 / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.eps_weight = torch.ger(eps_out, eps_in)
        self.eps_bias = eps_out

    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return torch.nn.functional.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)

        # Distributional dueling heads with NoisyLinear
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.num_atoms),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.num_actions * self.num_atoms),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # x: [B, C, H, W]
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        v = self.value_stream(features)                     # [B, N]
        a = self.advantage_stream(features)                 # [B, A*N]
        a = a.view(-1, self.num_actions, self.num_atoms)    # [B, A, N]
        v = v.unsqueeze(1)                                  # [B, 1, N]
        logits = v + (a - a.mean(dim=1, keepdim=True))      # [B, A, N]
        return logits  # logits per atom

    @staticmethod
    def probs_from_logits(logits):
        return torch.softmax(logits, dim=-1)  # [B, A, N]

    @staticmethod
    def expected_q_from_probs(probs, support):  # support: [1,1,N]
        return (probs * support).sum(dim=-1)  # [B, A]


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        stacked = np.stack(self.frames, axis=0).astype(np.float32) / 255.0
        return stacked

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0).astype(np.float32) / 255.0
        return stacked


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha # 控制優先級的使用程度 (0=純隨機, 1=完全按優先級)
        self.beta = beta # 控制重要性採樣的校正程度
        self.buffer = [None] * capacity # 使用 list 來存儲經驗
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        ########## YOUR CODE HERE (for Task 3) ########## 
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        transition = (state, action, reward, next_state, done)
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if self.size == 0:
            return [], [], []

        # 根據優先級計算抽樣機率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 根據機率進行加權抽樣，得到索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 計算重要性採樣權重 (IS weights)
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # 標準化權重
        
        # 根據索引取得經驗
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices, torch.tensor(weights, dtype=torch.float32)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # 根據新的 TD-Error 更新經驗的優先級
        # 加上一個極小值避免優先級為 0
        for i, error in zip(indices, errors):
            self.priorities[i] = error + 1e-5
        ########## END OF YOUR CODE (for Task 3) ########## 
    def __len__(self):
        return self.size
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.episode_losses = []
        input_shape = (4, 84, 84)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.num_atoms = 51
        self.v_min = -21.0
        self.v_max = 21.0
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device).view(1, 1, self.num_atoms)

        self.q_net = RainbowDQN(input_shape, self.num_actions, self.num_atoms).to(self.device)
        self.q_net.apply(init_weights)  # 只會影響 conv/linear；NoisyLinear 自行初始化
        self.target_net = RainbowDQN(input_shape, self.num_actions, self.num_atoms).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())



        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        
        self.n_step = 3 # for multi-step learning
        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.memory = PrioritizedReplayBuffer(args.memory_size)
        self.n_step_buffer = deque(maxlen=self.n_step)
        # PER β 退火：從 0.4 緩慢升到 1.0（約 1e6 次更新）
        self.beta_start = self.memory.beta
        self.beta_frames = 1_000_000
    def select_action(self, state):
        # if random.random() < self.epsilon:
        #     return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        self.q_net.reset_noise()
        with torch.no_grad():
            logits = self.q_net(state_tensor)  # [1, A, N]
            probs = RainbowDQN.probs_from_logits(logits)
            q_values = RainbowDQN.expected_q_from_probs(probs, self.support)  # [1, A]
        return q_values.argmax(dim=1).item()

    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            
            # state = obs task1 
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                reward = float(reward)
                done = terminated or truncated
                
                #next_state = next_obs
                next_state = self.preprocessor.step(next_obs)
                # self.memory.append((state, action, reward, next_state, done)) without prioritization
                # self.memory.add(state, action, reward, next_state, done)
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    n_step_reward = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(self.n_step)])
                    oldest_state, oldest_action, _, _, _ = self.n_step_buffer[0]
                    _, _, _, final_next_state, final_done = self.n_step_buffer[-1]
                    self.memory.add(oldest_state, oldest_action, n_step_reward, final_next_state, final_done)
                # Update the state for the next step
                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                if done:
                    while len(self.n_step_buffer) > 0:
                        n = len(self.n_step_buffer)
                        n_step_reward = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(n)])
                        oldest_state, oldest_action, _, _, _ = self.n_step_buffer[0]
                        _, _, _, final_next_state, final_done = self.n_step_buffer[-1]
                        
                        self.memory.add(oldest_state, oldest_action, n_step_reward, final_next_state, final_done)
                        # 移除最舊的，直到清空
                        self.n_step_buffer.popleft()
                #for 400k, 800k, 1.2M, 1.6M, and 1M environment steps.
                if step_count % 4000000 == 0 or step_count % 8000000 == 0 or step_count % 12000000 == 0 or step_count % 16000000 == 0 or step_count % 1000000 == 0:
                    model_path = os.path.join(self.save_dir, f"model_step{step_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    wandb.log({
                        "Episode": ep,
                        "Total Reward": total_reward,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 

            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })

            if self.episode_losses:  # 確保不是空 list 才計算 loss 統計
                avg_loss = sum(self.episode_losses) / len(self.episode_losses)
                max_loss = max(self.episode_losses)
                min_loss = min(self.episode_losses)
                std_loss = np.std(self.episode_losses)

                wandb.log({
                    "Episode": ep,
                    "Avg Loss": avg_loss,
                    "Max Loss": max_loss,
                    "Min Loss": min_loss,
                    "Std Loss": std_loss
                })

                self.episode_losses.clear()
            ########## END OF YOUR CODE ##########  
            if ep % 10 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 10 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        #state = obs
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.q_net(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                q_values = (probs * self.support).sum(dim=-1)  # [1, A]
                action = q_values.argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            #state = next_obs
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        #transitions = random.sample(self.memory, self.batch_size)
        transitions, indices, weights = self.memory.sample(self.batch_size) # For Task 3, use prioritized replay
        states, actions, rewards, next_states, dones = zip(*transitions)
      
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        # logits_current: [B, A, N]
        logits_current = self.q_net(states)
        probs_current = RainbowDQN.probs_from_logits(logits_current)
        q_current = RainbowDQN.expected_q_from_probs(probs_current, self.support)  # [B, A]

        # 取出當前動作對應的 logits: [B, N]
        batch_idx = torch.arange(self.batch_size, device=self.device)
        logits_a = logits_current[batch_idx, actions, :]  # [B, N]
        log_probs_a = torch.log_softmax(logits_a, dim=-1)  # [B, N]

        with torch.no_grad():
            # Double DQN: 用 online 決定 a*, 用 target 估計分佈
            next_logits_online = self.q_net(next_states)            # [B, A, N]
            next_probs_online = RainbowDQN.probs_from_logits(next_logits_online)
            next_q_online = RainbowDQN.expected_q_from_probs(next_probs_online, self.support)  # [B, A]
            next_actions = next_q_online.argmax(dim=1)              # [B]

            next_logits_target = self.target_net(next_states)       # [B, A, N]
            next_probs_target = torch.softmax(next_logits_target, dim=-1)  # [B, A, N]

            # 取出 a* 的分佈： [B, N]
            next_probs_star = next_probs_target[batch_idx, next_actions, :]

            # C51 投影
            Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * (1.0 - dones.unsqueeze(1)) * self.support.view(1, -1)
            Tz = Tz.clamp(self.v_min, self.v_max)

            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros_like(next_probs_star)  # [B, N]
            # 分配到 l 與 u
            offset = (torch.arange(self.batch_size, device=self.device) * self.num_atoms).unsqueeze(1)

            l_idx = (l + offset).view(-1)
            u_idx = (u + offset).view(-1)
            m_flat = m.view(-1)
            prob_flat = next_probs_star.view(-1)
            (m_flat.index_add_(0, l_idx, prob_flat * (u.float() - b).view(-1)))
            (m_flat.index_add_(0, u_idx, prob_flat * (b - l.float()).view(-1)))
            m = m_flat.view_as(m)  # [B, N]

        # Cross-entropy loss: - sum m * log p
        per_sample_loss = -(m * log_probs_a).sum(dim=1)  # [B]
        loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # PER 優先級更新：用期望值的 TD 誤差（分佈期望）
        with torch.no_grad():
            q_a_current = (torch.softmax(logits_a, dim=-1) * self.support.view(1, -1)).sum(dim=1)  # [B]
            q_a_target = (m * self.support.view(1, -1)).sum(dim=1)                                  # [B]
            td_errors = (q_a_target - q_a_current).abs().detach().cpu().numpy()

        self.memory.update_priorities(indices, td_errors)

        # β 退火
        self.memory.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.train_count / self.beta_frames))

        # 目標網路同步、日誌
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_a_current.mean().item():.3f} std: {q_a_current.std().item():.3f}")
        self.episode_losses.append(loss.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_curser")
    parser.add_argument("--wandb-run-name", type=str, default="pong-v5-rainbow-tuned")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=1000000) # Adjusted
    parser.add_argument("--lr", type=float, default=1e-4) # Adjusted
    parser.add_argument("--discount-factor", type=float, default=0.99)
    # Epsilon can be kept low or removed when using NoisyNets
    parser.add_argument("--epsilon-start", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=2000) # Adjusted
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=4000000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()
