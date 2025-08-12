import torch
import numpy as np
import random
import gymnasium as gym
import imageio
import ale_py
import os
from collections import deque
import argparse

# 匯入與訓練一致的網路與前處理
from LAB5_314551022_task3 import RainbowDQN, AtariPreprocessor

def evaluate_single_episode(env, preprocessor, model, support, device, seed, output_dir):
    obs, _ = env.reset(seed=seed)
    state = preprocessor.reset(obs)
    done = False
    total_reward = 0
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(state_tensor)                 # [1, A, N]
            probs = torch.softmax(logits, dim=-1)        # [1, A, N]
            q_values = (probs * support).sum(dim=-1)     # [1, A]
            action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = preprocessor.step(next_obs)

    out_path = os.path.join(output_dir, f"eval_seed{seed}.mp4")
    with imageio.get_writer(out_path, fps=30) as video:
        for f in frames:
            video.append_data(f)
    print(f"Seed {seed} → total reward {total_reward} → saved to {out_path}")
    return total_reward

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)
    num_atoms = 51
    v_min, v_max = -10.0, 10.0
    support = torch.linspace(v_min, v_max, num_atoms, device=device).view(1, 1, num_atoms)

    model = RainbowDQN(input_shape, num_actions, num_atoms=num_atoms).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    reward_list = []

    for i in range(args.episodes):
        seed = args.seed + i
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        total_reward = evaluate_single_episode(env, preprocessor, model, support, device, seed, args.output_dir)
        reward_list.append(total_reward)

    avg_reward = sum(reward_list) / len(reward_list)
    print(f"Average reward over {args.episodes} seeds ({args.seed}~{args.seed + args.episodes - 1}): {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task3")
    parser.add_argument("--episodes", type=int, default=20, help="Number of consecutive seeds to evaluate")
    parser.add_argument("--seed", type=int, default=186, help="Base seed for reproducibility")
    args = parser.parse_args()

    evaluate(args)
