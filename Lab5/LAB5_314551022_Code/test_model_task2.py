import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        # --- 卷積層 (Convolutional Layers) ---
        # 這些層負責從圖片中提取特徵
        self.conv = nn.Sequential(
            # 輸入: 4 x 84 x 84
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 輸出: 32 x 20 x 20
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 輸出: 64 x 9 x 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
            # 輸出: 64 x 7 x 7
        )
        
        # --- 全連接層 (Fully Connected Layers) ---
        # 這些層負責根據提取出的特徵來決定 Q-value
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # 讓輸入 x 依序通過卷積層和全連接層
        x = self.conv(x)
        x = x.view(x.size(0), -1) # <-- 把卷積層的輸出「拉平」成一維向量
        x = self.fc(x)
        return x

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.last_obs = None  # 新增：用於 max-over-two-frames

    def _max_two_frames(self, obs):
        """實作 max-over-two-frames 以減少閃爍效應"""
        if self.last_obs is None:
            return obs
        return np.maximum(obs, self.last_obs)

    def preprocess(self, obs):
        # 先做 max-over-two-frames
        obs_max = self._max_two_frames(obs)
        
        if len(obs_max.shape) == 3 and obs_max.shape[2] == 3:
            gray = cv2.cvtColor(obs_max, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs_max
        
        # 裁剪掉上下的分數區域（與訓練時保持一致）
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 更新 last_obs 以備下次使用
        self.last_obs = obs
        
        return resized.astype(np.uint8)

    def reset(self, obs):
        self.last_obs = None  # 重置
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked

def evaluate_single_seed(args, seed_value):
    """評估單一seed的函數"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定隨機種子
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

    # 建立環境
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(seed_value)
    env.observation_space.seed(seed_value)

    # 建立預處理器和模型
    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n
    input_shape = preprocessor.reset(env.reset()[0]).shape

    model = DQN(input_shape, num_actions).to(device)
    
    # 載入模型權重
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 執行單一episode
    obs, _ = env.reset(seed=seed_value)
    state = preprocessor.reset(obs)
    done = False
    total_reward = 0
    frames = []
    step_count = 0
    max_steps = args.max_steps

    while not done and step_count < max_steps:
        # 渲染當前幀
        frame = env.render()
        frames.append(frame)

        # 加入正規化，將輸入除以 255.0
        state_tensor = torch.from_numpy(state).float().div_(255.0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        # 執行動作
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = preprocessor.step(next_obs)
        step_count += 1

    # 儲存影片
    out_path = os.path.join(args.output_dir, f"eval_seed{seed_value}.mp4")
    try:
        with imageio.get_writer(out_path, fps=args.fps) as video:
            for f in frames:
                video.append_data(f)
        print(f"Seed {seed_value}: Reward = {total_reward}, Steps = {step_count} → Saved to {out_path}")
    except Exception as e:
        print(f"Error saving video for seed {seed_value}: {e}")
    
    env.close()
    return total_reward

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Base seed: {args.seed}")
    
    # 建立輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_rewards = []
    seeds_used = []
    
    # 生成20個連續種子以確保可重現性
    base_seed = args.seed
    print(f"Will use seeds: {base_seed} to {base_seed + 19}")
    
    for i in range(20):  # 固定20個seed
        seed_value = base_seed + i
        seeds_used.append(seed_value)
        
        print(f"Evaluating seed {i+1}/20: {seed_value}")
        reward = evaluate_single_seed(args, seed_value)
        
        if reward is not None:
            total_rewards.append(reward)
        else:
            print(f"Failed to evaluate seed {seed_value}")

    # 顯示統計結果
    print(f"\n=== Evaluation Summary ===")
    print(f"Total seeds evaluated: {len(total_rewards)}")
    print(f"Seeds used: {seeds_used}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Rewards: {total_rewards}")
    return np.mean(total_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task2", help="Output directory for videos")
    parser.add_argument("--seed", type=int, default=8382, help="Base random seed for evaluation")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output videos")
    
    args = parser.parse_args()
    rewards = evaluate(args)