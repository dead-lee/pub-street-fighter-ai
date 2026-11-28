"""用训练好的模型打《街头霸王 2》，并统计每一关的胜负。

有两种走法：

- 默认逐关读档：每一关都从自己的开局存档打起，关与关之间的地图不会出现。
- --no-load-level-saves：只读第 1 关存档。打赢后游戏自己飞到下一站，
  台词、世界地图、对决画面都由游戏内部播出来。输掉就停，不再读后面的存档。

加上 --realtime 时，画面全屏、按原速播放，并打开声音。
"""

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
LOAD_LEVEL_SAVES = True  # 默认每一关都读对应存档。
RESET_ROUND = False  # 打开后以一次击倒为结束，不再打三局两胜。
NUM_EPISODES = 10
SPEED = 4.0  # 快速预览时每 4 帧才画 1 帧。--realtime 会忽略它。


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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Play at original speed on a fullscreen window, with sound.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="With --realtime, also write the picture and sound to this mp4.",
    )
    return parser.parse_args()


def make_level_env(level: int, reset_round: bool,
                   rendering: bool, render_skip: int, realtime: bool = False,
                   record_path: Path | None = None):
    """建一局从指定关卡存档开始的环境。level 从 1 数起。"""
    env = make_sfighter_env(
        seed=level,
        obs_size=OBS_SIZE,
        render=rendering,
        initial_level=level - 1,
        time_penalty=TIME_PENALTY,
        render_skip=render_skip,
        reset_round=reset_round,
        auto_reset=False,
        realtime=realtime,
        record_path=None if record_path is None else str(record_path),
    )()
    return env


def play_level_from_save(level: int, model: PPO, reset_round: bool,
                         rendering: bool, render_skip: int,
                         realtime: bool = False) -> tuple[bool, float]:
    """读这一关自己的存档，打完一场就关掉环境。"""
    env = make_level_env(level, reset_round, rendering, render_skip, realtime)
    try:
        obs, info = env.reset(seed=level)
        victory, total_reward, _, _ = play_match(env, model, obs)
        return victory, total_reward
    finally:
        env.close()


def play_match(env, model: PPO, obs):
    """让模型一直打，直到这一场三局两胜结束，或步数被截断。"""
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
    # 对手血量已变为负数、自己还站着，并且不是因为超时被截断，才算赢。
    victory = (not truncated and info.get("enemy_hp", 0) < 0
               and info.get("agent_hp", -1) >= 0)
    return victory, total_reward, obs, info


def play_continuous_arcade_run(model: PPO, rendering: bool,
                               render_skip: int, realtime: bool = False,
                               record_path: Path | None = None) -> tuple[int, bool]:
    """从第 1 关存档打起，之后都靠游戏自己换对手，不再读档。

    不在第 12 场结束。木桶、四大天王和结局都在后面。
    窗口一直开着，直到用户按 Esc 或把窗口关掉。录像也在这时才收好。
    """
    env = make_level_env(1, False, rendering, render_skip, realtime, record_path)
    victories = 0
    run_complete = False
    try:
        # 整趟闯关只有这一次读档。后面的关卡、地图、结局都是游戏内部推进的。
        obs, info = env.reset(seed=1)
        level = 1
        while True:
            victory, reward, obs, info = play_match(env, model, obs)
            status = "Victory" if victory else "Defeat"
            print(f"Level {level}: {status}, reward={reward:.4f}", flush=True)
            if victory:
                victories += 1
            else:
                print("Match lost. The window stays open until it is closed.", flush=True)
            print("Advancing in the running game (no save-state reload)...", flush=True)
            advance = env.get_wrapper_attr("advance_to_next_match")
            # 不在结局处自己停。看的人按 Esc 或关掉窗口，录像才收好。
            obs, info = advance(max_steps=None)
            level += 1
        return victories, run_complete
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
    # 权重是按 84×84 的三通道运动图、27 个离散动作训练的，形状不对就不能测。
    if model.observation_space.shape != (3, OBS_SIZE, OBS_SIZE) or model.action_space != Discrete(27):
        raise ValueError("Model must use 84x84 motion RGB observations and 27 discrete actions")

    rendering = RENDERING and not args.no_render
    # 原速必须每帧都画、都出声，不能再用 --speed 跳帧。
    render_skip = 1 if args.realtime else max(1, round(args.speed))
    mode = (
        "matching save-state per level"
        if args.load_level_saves
        else "continuous arcade from Level 1"
    )
    if args.record is not None:
        args.record = args.record.expanduser()
    if args.realtime:
        print("Playback: original speed, fullscreen, sound. Press Esc to quit.", flush=True)
    if args.record is not None and not args.realtime:
        raise ValueError("--record needs --realtime so the picture and sound stay in step")
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
                model, rendering, render_skip, args.realtime, args.record
            )
            total_victories += victories
            # 输掉的那一关也算进分母，所以没打完时分母是胜场再加 1。
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
                    rendering, render_skip, args.realtime
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
