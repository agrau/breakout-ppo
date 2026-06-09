"""Smoke test: build full-game train+eval envs, load the expert model, run a
short learn. Validates the FULL_GAME pipeline before the real run."""
import config
import wrappers
from stable_baselines3 import PPO

if __name__ == "__main__":
    print("FULL_GAME =", config.FULL_GAME, "| N_ENVS =", config.N_ENVS,
          "| subproc =", config.USE_SUBPROC_ENV)
    env = wrappers.make_fullgame_train(config.N_ENVS, config.SEED,
                                       subproc=config.USE_SUBPROC_ENV)
    print("train obs space:", env.observation_space.shape)
    model = PPO.load("models/final_model.zip", env=env, device="cuda")
    print("loaded expert model, num_timesteps:", f"{model.num_timesteps:,}")
    model.learn(total_timesteps=2048, reset_num_timesteps=False, progress_bar=False)
    print("SHORT LEARN OK")
    env.close()
    ev = wrappers.make_fullgame_eval(config.SEED)
    print("eval obs space:", ev.observation_space.shape)
    ev.close()
    print("ALL TESTS PASSED")
