import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.aerial_manipulator import AerialManipulatorEnv

def main():
    print("Starting Phase 2: PPO Agent Training")
    
    # Ensure logs and models directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create vectorized environment for training
    env = make_vec_env(lambda: AerialManipulatorEnv(model_path="models/skydio_arm.xml", render_mode=None), n_envs=16)
    eval_env = make_vec_env(lambda: AerialManipulatorEnv(model_path="models/skydio_arm.xml", render_mode=None), n_envs=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/ppo_aerial_manipulator_tensorboard/",
        device="cuda:6"
    )
    
    # Train the agent deeply inside the accelerated hardware
    print("Training the model (Deep Run Phase 5)...")
    model.learn(total_timesteps=5000000, callback=eval_callback)
    
    # Save the final model
    model_path = "models/ppo_aerial_manipulator"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
