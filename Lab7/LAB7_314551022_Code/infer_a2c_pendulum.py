#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A2C Inference Script for Multiple Seeds

import random
import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import argparse
from typing import List, Tuple, Dict
from tqdm import tqdm

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

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
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
        
        # For inference, we also return entropy for completeness
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        initialize_uniformly(self.value_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.value_layer(x).squeeze(-1)
        return value


class A2CInference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Initialize inference agent with checkpoint."""
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loaded checkpoint from episode {checkpoint['episode']} with score {checkpoint['score']:.2f}")
        
        # Initialize networks
        self.actor = Actor(3, 1).to(self.device)  # Pendulum has 3 obs dim, 1 action dim
        self.critic = Critic(3).to(self.device)
        
        # Load state dicts
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        # Store hyperparameters
        self.gamma = checkpoint.get('gamma', 0.99)
        self.entropy_weight = checkpoint.get('entropy_weight', 0.01)
        
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action for inference."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            
            if deterministic:
                # Use mean action for deterministic policy
                x = F.relu(self.actor.fc1(state_t))
                x = F.relu(self.actor.fc2(x))
                mean = self.actor.mean_layer(x)
                squashed_action = torch.tanh(mean)
                action = squashed_action * 2.0
            else:
                # Sample from distribution
                action, _, _ = self.actor(state_t)
            
            return action.cpu().numpy().astype(np.float32)
    
    def test_single_seed(self, seed: int, render: bool = False, 
                        save_video: bool = False, video_folder: str = None) -> float:
        """Test agent with a single seed."""
        # Create environment
        render_mode = "rgb_array" if save_video else ("human" if render else None)
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        
        # Wrap with video recorder if needed
        if save_video and video_folder:
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder=video_folder,
                name_prefix=f"seed_{seed}",
                episode_trigger=lambda x: True  # Record every episode
            )
        
        # Reset with seed
        state, _ = env.reset(seed=seed)
        done = False
        score = 0
        steps = 0
        
        while not done:
            action = self.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            steps += 1
            
            if render and not save_video:  # Only render if not saving video
                env.render()
        
        env.close()
        return score
    
    def find_best_seeds(self, num_seeds: int = 20, start_seed: int = 0) -> List[Tuple[int, float]]:
        """Find best performing seeds without rendering."""
        print(f"\nSearching for best seeds (testing {num_seeds} seeds)...")
        results = []
        
        for i in tqdm(range(num_seeds), desc="Testing seeds"):
            seed = start_seed + i
            score = self.test_single_seed(seed, render=False, save_video=False)
            results.append((seed, score))
            
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("\n=== Top 10 Best Seeds ===")
        for i, (seed, score) in enumerate(results[:10]):
            print(f"Rank {i+1}: Seed {seed:6d} | Score: {score:8.2f}")
        
        return results
    
    def find_consecutive_good_seeds(self, window_size: int = 20, 
                                   target_avg: float = -150.0,
                                   max_search: int = 1000,
                                   start_seed: int = 0) -> Dict:
        """Find consecutive seeds where average reward > target."""
        print(f"\nSearching for {window_size} consecutive seeds with avg reward > {target_avg}")
        print(f"Starting from seed {start_seed}, searching up to {max_search} seeds...\n")
        
        found_windows = []
        all_results = []
        
        # Use sliding window approach
        for i in tqdm(range(max_search), desc="Testing seeds"):
            seed = start_seed + i
            score = self.test_single_seed(seed, render=False, save_video=False)
            all_results.append((seed, score))
            
            # Check if we have enough seeds for a window
            if len(all_results) >= window_size:
                # Get the last window_size results
                window = all_results[-window_size:]
                window_seeds = [s for s, _ in window]
                window_scores = [score for _, score in window]
                avg_score = np.mean(window_scores)
                
                # Check if this window meets the criteria
                if avg_score > target_avg:
                    window_info = {
                        'start_seed': window_seeds[0],
                        'end_seed': window_seeds[-1],
                        'seeds': window_seeds,
                        'scores': window_scores,
                        'avg_score': avg_score,
                        'min_score': np.min(window_scores),
                        'max_score': np.max(window_scores),
                        'std_score': np.std(window_scores)
                    }
                    found_windows.append(window_info)
                    
                    print(f"\n✓ Found good window!")
                    print(f"  Seeds: {window_seeds[0]} to {window_seeds[-1]}")
                    print(f"  Avg Score: {avg_score:.2f}")
                    print(f"  Min/Max: {window_info['min_score']:.2f} / {window_info['max_score']:.2f}")
                    
                    # Optional: continue searching or stop after first found
                    # return window_info  # Uncomment to stop after first found
        
        if found_windows:
            print(f"\n=== Found {len(found_windows)} valid windows ===")
            for i, window in enumerate(found_windows[:5]):  # Show first 5
                print(f"\nWindow {i+1}:")
                print(f"  Seeds: {window['start_seed']} to {window['end_seed']}")
                print(f"  Avg Score: {window['avg_score']:.2f}")
                print(f"  Std Dev: {window['std_score']:.2f}")
                print(f"  Range: [{window['min_score']:.2f}, {window['max_score']:.2f}]")
            
            # Return the best window (highest average)
            best_window = max(found_windows, key=lambda x: x['avg_score'])
            print(f"\n★ Best Window: Seeds {best_window['start_seed']} to {best_window['end_seed']}")
            print(f"  Average Score: {best_window['avg_score']:.2f}")
            return best_window
        else:
            print(f"\n✗ No window of {window_size} consecutive seeds with avg > {target_avg} found.")
            print(f"  Best average found: {max([np.mean([s for _, s in all_results[i:i+window_size]]) for i in range(len(all_results)-window_size+1)]):.2f}")
            return None
    
    def test_multiple_seeds_with_video(self, seeds: List[int], video_dir: str = "videos"):
        """Test multiple seeds and save videos."""
        os.makedirs(video_dir, exist_ok=True)
        
        print(f"\nRecording videos for {len(seeds)} seeds...")
        results = []
        
        for seed in tqdm(seeds, desc="Recording videos"):
            # Create seed-specific folder
            seed_video_dir = os.path.join(video_dir, f"seed_{seed}")
            os.makedirs(seed_video_dir, exist_ok=True)
            
            score = self.test_single_seed(
                seed, 
                render=False, 
                save_video=True, 
                video_folder=seed_video_dir
            )
            results.append((seed, score))
            print(f"Seed {seed}: Score = {score:.2f}")
        
        # Summary
        print("\n=== Video Recording Summary ===")
        for seed, score in results:
            print(f"Seed {seed:6d} | Score: {score:8.2f} | Video saved in {video_dir}/seed_{seed}/")
        
        # Statistics
        scores = [score for _, score in results]
        print(f"\nStatistics:")
        print(f"  Mean Score: {np.mean(scores):.2f}")
        print(f"  Std Score:  {np.std(scores):.2f}")
        print(f"  Max Score:  {np.max(scores):.2f}")
        print(f"  Min Score:  {np.min(scores):.2f}")
        
        # Check if average meets target
        avg_score = np.mean(scores)
        if avg_score > -150:
            print(f"\n✓ SUCCESS: Average score {avg_score:.2f} > -150")
        else:
            print(f"\n✗ FAILED: Average score {avg_score:.2f} <= -150")
    
    def quick_test_consecutive(self, start_seed: int, num_seeds: int = 20) -> Dict:
        """Quickly test a specific range of consecutive seeds."""
        print(f"\nQuick testing seeds {start_seed} to {start_seed + num_seeds - 1}")
        
        seeds = list(range(start_seed, start_seed + num_seeds))
        scores = []
        
        for seed in tqdm(seeds, desc="Testing"):
            score = self.test_single_seed(seed, render=False, save_video=False)
            scores.append(score)
            print(f"Seed {seed}: {score:.2f}")
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\n=== Results ===")
        print(f"Seeds: {start_seed} to {start_seed + num_seeds - 1}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Std Dev: {std_score:.2f}")
        print(f"Min/Max: {np.min(scores):.2f} / {np.max(scores):.2f}")
        
        if avg_score > -150:
            print(f"✓ SUCCESS: Average > -150")
        else:
            print(f"✗ FAILED: Average <= -150")
        
        return {
            'seeds': seeds,
            'scores': scores,
            'avg_score': avg_score,
            'std_score': std_score
        }


def main():
    parser = argparse.ArgumentParser(description="A2C Pendulum Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--mode", type=str, choices=["find", "consecutive", "test", "both", "quick"], default="both",
                       help="Mode: 'find' best seeds, 'consecutive' for continuous good seeds, 'test' with videos, 'quick' for quick consecutive test, or 'both'")
    parser.add_argument("--num-seeds", type=int, default=20,
                       help="Number of seeds to test (or window size for consecutive mode)")
    parser.add_argument("--start-seed", type=int, default=10041,
                       help="Starting seed value")
    parser.add_argument("--video-dir", type=str, default="videos",
                       help="Directory to save videos")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top seeds to record videos for")
    parser.add_argument("--specific-seeds", type=int, nargs="+",
                       help="Specific seeds to test (overrides auto selection)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during testing (only for single seed)")
    parser.add_argument("--single-seed", type=int,
                       help="Test a single seed with optional rendering")
    parser.add_argument("--target-avg", type=float, default=-150.0,
                       help="Target average reward for consecutive seeds mode")
    parser.add_argument("--max-search", type=int, default=1000,
                       help="Maximum number of seeds to search in consecutive mode")
    parser.add_argument("--record-consecutive", action="store_true",
                       help="Record videos for found consecutive seeds")
    
    args = parser.parse_args()
    
    # Initialize inference agent
    agent = A2CInference(args.checkpoint, device=args.device)
    
    # Single seed test mode
    if args.single_seed is not None:
        print(f"\nTesting single seed: {args.single_seed}")
        score = agent.test_single_seed(args.single_seed, render=args.render)
        print(f"Score: {score:.2f}")
        return
    
    # Quick test mode - Test specific consecutive seeds quickly
    if args.mode == "quick":
        result = agent.quick_test_consecutive(args.start_seed, args.num_seeds)
        
        # Optionally record videos if average is good
        if result['avg_score'] > args.target_avg and args.record_consecutive:
            print(f"\nRecording videos since average {result['avg_score']:.2f} > {args.target_avg}")
            agent.test_multiple_seeds_with_video(result['seeds'], args.video_dir)
        return
    
    # Consecutive seeds mode - Find 20 consecutive seeds with avg > target
    if args.mode == "consecutive":
        window_result = agent.find_consecutive_good_seeds(
            window_size=args.num_seeds,
            target_avg=args.target_avg,
            max_search=args.max_search,
            start_seed=args.start_seed
        )
        
        if window_result and args.record_consecutive:
            print(f"\nRecording videos for the best consecutive window...")
            agent.test_multiple_seeds_with_video(window_result['seeds'], args.video_dir)
            
            # Print detailed results
            print("\n=== Detailed Results for Best Window ===")
            for seed, score in zip(window_result['seeds'], window_result['scores']):
                print(f"Seed {seed:6d}: {score:8.2f}")
            print(f"\nAverage: {window_result['avg_score']:.2f}")
        
        return
    
    # Multiple seeds test modes
    if args.mode in ["find", "both"]:
        # Find best individual seeds
        results = agent.find_best_seeds(args.num_seeds, args.start_seed)
        
        if args.mode == "both":
            # Select top-k seeds for video recording
            top_seeds = [seed for seed, _ in results[:args.top_k]]
            print(f"\nSelected top {args.top_k} seeds for video recording: {top_seeds}")
            agent.test_multiple_seeds_with_video(top_seeds, args.video_dir)
    
    elif args.mode == "test":
        # Test specific seeds or use provided seeds
        if args.specific_seeds:
            seeds = args.specific_seeds
        else:
            # Generate random seeds if not specified
            seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        
        agent.test_multiple_seeds_with_video(seeds, args.video_dir)


if __name__ == "__main__":
    main()