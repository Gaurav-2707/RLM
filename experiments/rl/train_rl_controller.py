"""
Train the RL Controller using Offline DQN on generated JSON traces.

This script loads traces from a specified directory, validates that the
agent can overfit on a small subset (memorization check), and then
trains the full DQN agent, saving the weights for the inference pipeline.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Try to import from local package, assuming run from repo root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from RLM.experiments.rl.offline_env import ReasoningOvershootEnv, load_traces_from_dir


def test_overfitting(traces_subset):
    """
    Mandatory verification step:
    Ensure DQN can memorize 10 traces perfectly. If it can't, the state
    representation or reward formulation is broken.
    """
    print("\n" + "="*50)
    print("MANDATORY VERIFICATION: Overfit Test (10 traces)")
    print("="*50)
    
    # Create a small environment with just 10 traces
    small_env = ReasoningOvershootEnv(traces_subset, continuation_penalty=-0.03)
    
    model = DQN("MlpPolicy", small_env, verbose=0, learning_rate=1e-3, exploration_fraction=0.1)
    
    # Train heavily to force memorization
    model.learn(total_timesteps=5000, progress_bar=False)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, small_env, n_eval_episodes=10, deterministic=True)
    
    print(f"Overfit Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # A perfectly overfit model on traces where a correct answer exists should achieve > 0 reward.
    # If the reward is heavily negative, it failed to learn.
    if mean_reward < 0.0:
        print("WARNING: Model failed to memorize the 10 traces (negative mean reward).")
        print("Check reward formulation and state features.")
        return False
    else:
        print("SUCCESS: Model successfully memorized the subset.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="RLM/experiments/results/scaling_law_ollama_llama3.1_8b_gsm8k")
    parser.add_argument("--out_path", type=str, default="RLM/weights/dqn_controller.zip")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--skip_overfit", action="store_true", help="Skip the 10-trace overfit test")
    args = parser.parse_args()
    
    print(f"Loading traces from {args.trace_dir}...")
    traces = load_traces_from_dir(args.trace_dir)
    
    if len(traces) == 0:
        print(f"Error: No traces found in {args.trace_dir}.")
        print("You must run the fixed_budget sweep first to generate data!")
        return
        
    print(f"Loaded {len(traces)} traces.")
    
    if not args.skip_overfit and len(traces) >= 10:
        success = test_overfitting(traces[:10])
        if not success:
            print("Aborting full training due to overfit test failure.")
            return

    print("\n" + "="*50)
    print("TRAINING FULL RL CONTROLLER")
    print("="*50)
    
    env = ReasoningOvershootEnv(traces, continuation_penalty=-0.03)
    
    # Initialize DQN
    # Use small network, low learning rate, aggressive exploration initially
    model = DQN("MlpPolicy", env, verbose=1, 
                learning_rate=5e-4, 
                buffer_size=10000,
                exploration_fraction=0.2,
                policy_kwargs=dict(net_arch=[64, 64]))
    
    # Train
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    
    # Create out dir
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    # Save weights
    model.save(args.out_path)
    print(f"Model saved to {args.out_path}")
    
    # Evaluate on the training set (offline RL is usually just mimicking the best policy found)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=min(100, len(traces)), deterministic=True)
    print(f"Final Evaluation Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
