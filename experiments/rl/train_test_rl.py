import os
import argparse
from stable_baselines3 import PPO
from RLM.experiments.rl.test_driven_env import TestDrivenEnv

def main():
    parser = argparse.ArgumentParser(description="Train Test-Driven RL Agent")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to the repository to test")
    parser.add_argument("--test_command", type=str, default="pytest", help="Command to run tests")
    parser.add_argument("--reward_type", type=str, default="both", choices=["binary", "granular", "both"])
    parser.add_argument("--timesteps", type=int, default=50)
    args = parser.parse_args()

    print(f"Initializing Test-Driven RL environment for repo: {args.repo_path}")
    print(f"Test command: {args.test_command}")
    print(f"Reward type: {args.reward_type}")
    
    # Initialize Environment
    env = TestDrivenEnv(
        repo_path=args.repo_path,
        test_command=args.test_command,
        reward_type=args.reward_type,
        max_steps=20,
        inject_synthetic_bug=True,
    )
    
    # Use PPO for training
    print("Initializing PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2, batch_size=2)
    
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    
    print("Training complete. Saving model...")
    os.makedirs("weights", exist_ok=True)
    model.save("weights/test_driven_ppo")
    print("Model saved to RLM/weights/test_driven_ppo.zip")

if __name__ == "__main__":
    main()
