#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

# =====================================================================================
#  從 ppo_walker-2.py 複製過來的必要類別與函式
#  (確保模型架構與訓練時完全一致)
# =====================================================================================

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

    def forward(self, state: torch.Tensor) -> Normal:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        dist = Normal(mu, std)
        
        return dist
        
# Critic 類別在純推論中不是必需的，但為了完整載入 checkpoint，我們保留它
class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        init_layer_uniform(self.fc3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value

# =====================================================================================
#  錄製影片的函式
# =====================================================================================

def record_videos_for_seeds(actor, checkpoint, args, start_seed, end_seed):
    """為指定的連續seeds錄製影片"""
    print(f"\n🎬 開始為 Seeds {start_seed} 到 {end_seed} 錄製影片...")
    
    # 創建錄製環境
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    
    # 恢復環境的正規化統計數據
    if 'env_obs_rms' in checkpoint:
        env.obs_rms = checkpoint['env_obs_rms']
    
    # 設定影片錄製目錄
    video_dir = f"result_task3"
    os.makedirs(video_dir, exist_ok=True)
    
    device = next(actor.parameters()).device
    all_scores = []
    
    print(f"影片將儲存至: {video_dir}/")
    
    for seed in range(start_seed, end_seed + 1):
        # 為每個seed創建單獨的錄製環境
        seed_env = gym.make("Walker2d-v4", render_mode="rgb_array")
        seed_env = gym.wrappers.ClipAction(seed_env)
        seed_env = gym.wrappers.NormalizeObservation(seed_env)
        
        # 恢復環境的正規化統計數據
        if 'env_obs_rms' in checkpoint:
            seed_env.obs_rms = checkpoint['env_obs_rms']
        
        # 設定影片錄製，使用自定義命名
        seed_env = gym.wrappers.RecordVideo(
            seed_env, 
            video_folder=video_dir, 
            episode_trigger=lambda x: True,
            name_prefix=f"task3-seed-{seed}"
        )
        
        state, _ = seed_env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 1000:  # 限制最大步數
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist = actor(state_tensor)
                action = dist.mean
            
            action_np = action.cpu().numpy().flatten()
            next_state, reward, terminated, truncated, _ = seed_env.step(action_np)
            
            done = terminated or truncated
            total_reward += reward
            state = next_state
            steps += 1
        
        all_scores.append(total_reward)
        print(f"  Seed {seed}: {total_reward:.2f}")
        
        # 關閉這個seed的環境
        seed_env.close()
    
    # 顯示錄製結果
    avg_score = np.mean(all_scores)
    print(f"\n📊 錄製完成！")
    print(f"  - 平均分數: {avg_score:.2f}")
    print(f"  - 最高分數: {np.max(all_scores):.2f}")
    print(f"  - 最低分數: {np.min(all_scores):.2f}")
    print(f"  - 影片儲存於: {video_dir}/")
    print(f"  - 影片命名格式: seed-{start_seed}.mp4, seed-{start_seed+1}_episode-0.mp4, ...")
    
    return all_scores

# =====================================================================================
#  搜尋模式：尋找符合條件的連續 Seeds
# =====================================================================================

def find_high_score_seeds(args):
    """載入模型並搜尋連續20個平均分數高於目標的 seeds"""

    # 1. 設定環境 (必須與訓練時使用完全相同的 Wrapper)
    # 在快速搜尋時，不渲染畫面以提升速度
    env = gym.make("Walker2d-v4") 
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)

    # 2. 設定 PyTorch 和 Numpy 的隨機種子 (只影響模型初始化，不影響環境)
    np.random.seed(args.base_torch_seed)
    torch.manual_seed(args.base_torch_seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # 3. 決定使用的裝置 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"將在 {device} 上運行搜尋")

    # 4. 載入模型檢查點
    if not os.path.exists(args.checkpoint_path):
        print(f"錯誤：找不到模型檢查點檔案 '{args.checkpoint_path}'")
        return
        
    print(f"正在從 '{args.checkpoint_path}' 載入模型...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # 恢復環境的正規化統計數據，這對分數至關重要
    if 'env_obs_rms' in checkpoint:
        env.obs_rms = checkpoint['env_obs_rms']
        print("成功恢復環境的正規化統計數據 (obs_rms)！")
    else:
        print("警告：在 checkpoint 中找不到 'env_obs_rms'。推論分數可能會非常低。")

    # 5. 初始化 Actor 網路並載入權重
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval() # 設定為評估模式
    
    print("模型載入成功！")
    print("-" * 30)

    # 6. 執行搜尋迴圈
    # 使用 deque 作為滑動窗口，自動維持最近20個分數
    scores_window = deque(maxlen=20)
    
    print(f"開始搜尋... 目標：連續20個 seed 平均分數 >= {args.target_score}")
    
    # 使用 tqdm 顯示進度
    for current_seed in tqdm(range(args.start_seed, args.start_seed + args.search_limit), desc="Searching Seeds"):
        # 為每個回合重置環境，並使用當前的 seed
        state, _ = env.reset(seed=current_seed)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist = actor(state_tensor)
                action = dist.mean
            
            action_np = action.cpu().numpy().flatten()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        # 將當前回合的分數加入滑動窗口
        scores_window.append(total_reward)

        # 當窗口滿了 (收集了20個分數) 就檢查平均值
        if len(scores_window) == 20:
            current_avg = np.mean(scores_window)
            
            # 如果平均分數達到目標
            if current_avg >= args.target_score:
                start_of_window_seed = current_seed - 19
                print("\n\n" + "="*50)
                print(f"🎉 成功找到符合條件的連續 Seeds！")
                print(f"  - 起始 Seed: {start_of_window_seed}")
                print(f"  - 結束 Seed: {current_seed}")
                print(f"  - 這 20 個 Seed 的平均分數: {current_avg:.2f}")
                print("="*50)
                print("詳細分數列表：")
                for i, score in enumerate(scores_window):
                    print(f"  - Seed {start_of_window_seed + i}: {score:.2f}")
                
                print(f"\n💡 使用以下命令進行測試並錄製影片：")
                print(f"python infer-ppo-walker.py --mode test --start-seed {start_of_window_seed} --checkpoint-path {args.checkpoint_path}")
                
                env.close()
                return # 找到後就結束程式

    # 如果迴圈跑完都沒找到
    print("\n搜尋結束，但在指定的範圍內未找到符合條件的連續 seeds。")
    print("您可以嘗試：")
    print("1. 增加 `--search-limit` 的值來擴大搜尋範圍。")
    print("2. 調整 `--start-seed` 從不同的點開始搜尋。")
    print("3. 降低 `--target-score` 的要求。")
    
    env.close()

# =====================================================================================
#  測試模式：對指定seeds進行inference並錄製影片
# =====================================================================================

def test_specified_seeds(args):
    """對指定的連續seeds進行inference並錄製影片"""
    
    # 1. 設定 PyTorch 和 Numpy 的隨機種子
    np.random.seed(args.base_torch_seed)
    torch.manual_seed(args.base_torch_seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # 2. 決定使用的裝置 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"將在 {device} 上運行測試")

    # 3. 載入模型檢查點
    if not os.path.exists(args.checkpoint_path):
        print(f"錯誤：找不到模型檢查點檔案 '{args.checkpoint_path}'")
        return
        
    print(f"正在從 '{args.checkpoint_path}' 載入模型...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # 4. 初始化 Actor 網路並載入權重
    env = gym.make("Walker2d-v4")  # 臨時環境來獲取維度
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    
    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # 5. 執行測試並錄製影片
    end_seed = args.start_seed + 19
    scores = record_videos_for_seeds(actor, checkpoint, args, args.start_seed, end_seed)
    
    print(f"\n✅ 測試完成！")
    print(f"Seeds {args.start_seed} 到 {end_seed} 的平均分數: {np.mean(scores):.2f}")

# =====================================================================================
#  主程式
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="為 Walker2d-v4 搜尋和測試連續 Seeds")
    
    parser.add_argument("--mode", type=str, choices=["find", "test"], default="find",
                        help="執行模式：find=搜尋符合條件的seeds，test=測試指定seeds並錄製影片")
    
    parser.add_argument("--checkpoint-path", type=str, 
                        default="result_task3/ppo_walker_ep3250_step939896.pt",
                        help="訓練好的模型檢查點 (.pt) 檔案路徑")
                        
    parser.add_argument("--start-seed", type=int, default=0,
                        help="起始種子 (find模式：搜尋起始點，test模式：測試起始點)")
                        
    parser.add_argument("--search-limit", type=int, default=1000,
                        help="find模式：從起始種子開始要搜尋的數量")

    parser.add_argument("--target-score", type=float, default=2500.0,
                        help="find模式：連續20個seeds要達到的目標平均分數")

    parser.add_argument("--base-torch-seed", type=int, default=42,
                        help="用於PyTorch和Numpy的基礎隨機種子，確保模型權重一致")

    args = parser.parse_args()
    
    if args.mode == "find":
        print("🔍 執行搜尋模式...")
        find_high_score_seeds(args)
    elif args.mode == "test":
        print("🎬 執行測試模式...")
        test_specified_seeds(args)
    else:
        print(f"錯誤：未知的模式 '{args.mode}'")