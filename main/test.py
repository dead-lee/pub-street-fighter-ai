import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from gymnasium.spaces import Discrete

from main.env_utils import make_sfighter_env, validate_rom
from main.train import detect_device

import torch


DEFAULT_MODEL_PATH = Path(
    "logs/trained_models/ppo_ryu.zip"
)
TIME_PENALTY = 0.0005
OBS_SIZE = 84
RENDERING = True
LOAD_LEVEL_SAVES = True  # Reset from each level's save-state.
RESET_ROUND = False  # End the episode at each KO when enabled.
NUM_EPISODES = 10
SPEED = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the single PPO model used for all 12 levels.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help="Number of evaluation passes across Levels 1-12.",
    )
    parser.add_argument(
        "--load-level-saves",
        action=argparse.BooleanOptionalAction,
        default=LOAD_LEVEL_SAVES,
        help=(
            "Load each level's matching save-state (default); use "
            "--no-load-level-saves for one continuous arcade run."
        ),
    )
    parser.add_argument(
        "--reset-round", action=argparse.BooleanOptionalAction,
        default=RESET_ROUND,
        help=(
            "Treat one KO as the result instead of playing the full "
            "best-of-three match."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed", type=float, default=SPEED)
    return parser.parse_args()


def make_level_env(level: int, reset_round: bool,
                   rendering: bool, render_skip: int):
    env = make_sfighter_env(
        seed=level,
        obs_size=OBS_SIZE,
        render=rendering,
        initial_level=level - 1,
        time_penalty=TIME_PENALTY,
        render_skip=render_skip,
        reset_round=reset_round,
        auto_reset=False,
    )()
    return env


def play_level_from_save(level: int, model: PPO, reset_round: bool,
                         rendering: bool, render_skip: int) -> tuple[bool, float]:
    env = make_level_env(level, reset_round, rendering, render_skip)
    try:
        obs, info = env.reset(seed=level)
        victory, total_reward, _, _ = play_match(env, model, obs)
        return victory, total_reward
    finally:
        env.close()


def play_match(env, model: PPO, obs):
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
    victory = (not truncated and info.get("enemy_hp", 0) < 0
               and info.get("agent_hp", -1) >= 0)
    return victory, total_reward, obs, info


def play_continuous_arcade_run(model: PPO, rendering: bool,
                               render_skip: int) -> tuple[int, bool]:
    env = make_level_env(1, False, rendering, render_skip)
    victories = 0
    try:
        # This is the only save-state load in a continuous arcade run.
        obs, info = env.reset(seed=1)
        for level in range(1, 13):
            victory, reward, obs, info = play_match(env, model, obs)
            status = "Victory" if victory else "Defeat"
            print(f"Level {level}: {status}, reward={reward:.4f}")
            if not victory:
                return victories, False
            victories += 1
            if level < 12:
                print("Advancing in the running game (no save-state reload)...")
                advance = env.get_wrapper_attr("advance_to_next_match")
                obs, info = advance()
        return victories, True
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.speed < 1:
        raise ValueError("--speed must be at least 1")
    if not args.load_level_saves and args.reset_round:
        raise ValueError(
            "--reset-round cannot be combined with "
            "--no-load-level-saves because a KO is not a completed match"
        )

    if args.model_path.suffix != ".zip":
        raise ValueError("--model-path must point to a .zip model file")

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    validate_rom()
    model = PPO.load(str(args.model_path), device=detect_device(args.device))
    if model.observation_space.shape != (3, OBS_SIZE, OBS_SIZE) or model.action_space != Discrete(27):
        raise ValueError("Model must use 84x84 motion RGB observations and 27 discrete actions")

    rendering = RENDERING and not args.no_render
    render_skip = max(1, round(args.speed))
    mode = (
        "matching save-state per level"
        if args.load_level_saves
        else "continuous arcade from Level 1"
    )
    match_format = "single KO" if args.reset_round else "best-of-three"
    print(
        f"Testing {args.episodes} pass(es), levels 1-12, "
        f"mode={mode}, format={match_format}"
    )

    total_victories = 0
    total_levels = 0
    complete_runs = 0
    for episode in range(1, args.episodes + 1):
        pass_name = "Save-state pass" if args.load_level_saves else "Arcade run"
        print(f"\n=== {pass_name} {episode}/{args.episodes} ===")
        if not args.load_level_saves:
            victories, run_complete = play_continuous_arcade_run(
                model, rendering, render_skip
            )
            total_victories += victories
            total_levels += victories + (not run_complete)
        else:
            run_complete = True
            for level in range(1, 13):
                print(
                    f"Level {level}: loading save and using "
                    f"{args.model_path.name}"
                )
                victory, reward = play_level_from_save(
                    level, model, args.reset_round,
                    rendering, render_skip
                )
                total_levels += 1
                status = "Victory" if victory else "Defeat"
                print(f"Level {level}: {status}, reward={reward:.4f}")
                if victory:
                    total_victories += 1
                else:
                    run_complete = False
        if run_complete:
            complete_runs += 1
            print("All 12 levels won.")
        else:
            if args.load_level_saves:
                print("Evaluation pass complete: one or more levels lost.")
            else:
                print("Arcade run stopped at the first defeated level.")
            if episode < args.episodes and not args.load_level_saves:
                print("Starting a new arcade attempt from Level 1...")

    print(
        f"\nLevel win rate: {total_victories / total_levels:.3f} "
        f"({total_victories}/{total_levels})"
    )
    completion_label = (
        "All-level pass rate"
        if args.load_level_saves
        else "Full arcade clear rate"
    )
    print(
        f"{completion_label}: {complete_runs / args.episodes:.3f} "
        f"({complete_runs}/{args.episodes})"
    )


if __name__ == "__main__":
    main()
