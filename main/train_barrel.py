"""用和对战相同的 PPO、画面和网络，只在滚木桶上训练。

出招锁成轻拳升龙。模型只决定往左走、往右走、站着等，还是朝哪边发这一招。
奖励是游戏实际加上的分数，所以全破和时间奖励都算在目标里。
升龙打空要扣分。人还没站稳就按，会被吃掉。
从第 2 秒起，一整秒里没有打出伤害，再扣掉相当于一只桶的奖励。
和上一发升龙朝向相反、而且打中的那一下，分数再给一倍。
对战默认 12 关各 64 个环境。木桶只有这一关，默认 8 个并行模拟器，把核留给网络更新。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


FEATURES_DIM = 1024
DEFAULT_N_STEPS = 128
DEFAULT_N_ENVS = 8
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_EPOCHS = 3
TIME_PENALTY = 0.0005
DEFAULT_LOG_DIR = "logs/BARREL_MOTION"
# 固定套路目前大约能拿到的分数。评估超过它就另存一份。
SCRIPT_SCORE = 30300


def detect_device(requested: str) -> str:
    """auto 时按 CUDA、Apple MPS、CPU 的顺序挑一个能用的设备。"""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    # 对战一局很长，0.01 的熵系数用来防止过早只剩一招。
    # 木桶只有几百步，这个系数会把策略留在乱按附近，高分招式留不下来。
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--features-dim", type=int, default=FEATURES_DIM)
    parser.add_argument("--time-penalty", type=float, default=TIME_PENALTY)
    parser.add_argument("--checkpoint-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument(
        "--eval-only", type=Path,
        help="只评估这份模型，不训练。打印每局实际加上的分数。",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="评估时全屏、原速、有声音。只和 --eval-only 一起用。",
    )
    return parser


def ppo_hyperparams(args: argparse.Namespace) -> dict[str, object]:
    """新建和续训用同一组超参，和学习率的设置方式跟对战训练一致。"""
    return {
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "target_kl": args.target_kl,
        "max_grad_norm": 0.5,
    }


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "total-timesteps": args.total_timesteps,
        "num-envs": args.num_envs,
        "n-steps": args.n_steps,
        "batch-size": args.batch_size,
        "n-epochs": args.n_epochs,
        "learning-rate": args.learning_rate,
        "features-dim": args.features_dim,
        "checkpoint-timesteps": args.checkpoint_timesteps,
        "eval-episodes": args.eval_episodes,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.time_penalty < 0:
        raise ValueError("--time-penalty must be non-negative")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("gamma and gae-lambda must be in (0, 1]")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    rollout = args.num_envs * args.n_steps
    if args.batch_size > rollout or rollout % args.batch_size:
        raise ValueError(
            f"--batch-size must divide and not exceed rollout size {rollout}"
        )
    for path in (args.resume_model, args.eval_only):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")


def play_barrel_episode(model: PPO, env) -> int:
    """打完一局木桶，返回游戏实际加上的分数。"""
    obs, _ = env.reset()
    done = False
    gained = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        gained = int(info.get("barrel_gained", gained))
        done = bool(terminated or truncated)
    return gained


class BarrelScoreCallback(BaseCallback):
    """隔一段用当前策略打几局，记下实际得分。比固定套路高就另存。"""

    def __init__(self, eval_env, eval_every: int, eval_episodes: int, best_path: Path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.best_path = best_path
        self.best_score = -1

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_every != 0:
            return True
        scores = [
            play_barrel_episode(self.model, self.eval_env)
            for _ in range(self.eval_episodes)
        ]
        mean_score = sum(scores) / len(scores)
        print(
            f"Barrel eval at {self.num_timesteps:,}: "
            f"scores={scores} mean={mean_score:.0f}",
            flush=True,
        )
        if mean_score > self.best_score:
            self.best_score = mean_score
            self.model.save(self.best_path)
            print(f"New best barrel model: {self.best_path}.zip ({mean_score:.0f})", flush=True)
        return True


def main(argv: Sequence[str] | None = None) -> int:
    from main.env_utils import make_barrel_env, validate_rom

    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = detect_device(args.device)
    validate_rom()

    if args.watch and not args.eval_only:
        raise ValueError("--watch only applies to --eval-only")
    if args.eval_only:
        env = make_barrel_env(
            seed=args.seed,
            time_penalty=args.time_penalty,
            render=args.watch,
            realtime=args.watch,
        )()
        try:
            model = PPO.load(str(args.eval_only), device=device)
            scores = [
                play_barrel_episode(model, env) for _ in range(args.eval_episodes)
            ]
        finally:
            env.close()
        print(
            f"scores={scores} mean={sum(scores) / len(scores):.0f} "
            f"script_baseline={SCRIPT_SCORE}",
            flush=True,
        )
        return 0

    from main.custom_wrappers import BARREL_SCORE_SCALE, BARREL_TURN_BONUS

    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "TRAINING"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **{key: (str(value) if isinstance(value, Path) else value)
           for key, value in vars(args).items()},
        "device_resolved": device,
        "observation_mode": "motion_rgb",
        "actions": "wait_left_right_light_shoryuken",
        "reward": (
            f"score_delta * {BARREL_SCORE_SCALE:g} - time_penalty "
            "- whiff - wrong_move - blocked - idle_second "
            f"+ opposite_shoryuken * {BARREL_TURN_BONUS:g}"
        ),
    }
    (log_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    env = SubprocVecEnv(
        [
            make_barrel_env(
                seed=args.seed + worker,
                use_monitor=True,
                time_penalty=args.time_penalty,
            )
            for worker in range(args.num_envs)
        ]
    )
    eval_env = make_barrel_env(
        seed=args.seed + 10_000,
        time_penalty=args.time_penalty,
    )()
    model: PPO | None = None
    completed = False
    try:
        hyperparams = ppo_hyperparams(args)
        if args.resume_model:
            print(f"Continuing barrel model: {args.resume_model}", flush=True)
            model = PPO.load(
                str(args.resume_model),
                env=env,
                device=device,
                custom_objects=hyperparams,
                print_system_info=False,
            )
            reset_num_timesteps = False
        else:
            model = PPO(
                "CnnPolicy",
                env,
                device=device,
                verbose=1,
                seed=args.seed,
                policy_kwargs={
                    "features_extractor_kwargs": {
                        "features_dim": args.features_dim,
                    }
                },
                **hyperparams,
            )
            reset_num_timesteps = True
        best_path = checkpoint_dir / "ppo_barrel_ryu_best"
        callbacks = [
            CheckpointCallback(
                save_freq=max(args.checkpoint_timesteps // args.num_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_barrel_ryu",
            ),
            BarrelScoreCallback(
                eval_env,
                eval_every=args.checkpoint_timesteps,
                eval_episodes=args.eval_episodes,
                best_path=best_path,
            ),
        ]
        print(
            f"Barrel PPO on {device}: {args.num_envs} envs, "
            f"rollout={args.num_envs * args.n_steps}, "
            f"features={args.features_dim}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_Barrel",
            reset_num_timesteps=reset_num_timesteps,
        )
        completed = True
    except KeyboardInterrupt:
        if model is not None:
            interrupted = checkpoint_dir / "ppo_barrel_ryu_interrupted"
            model.save(interrupted)
            print(f"Interrupted barrel model saved to: {interrupted}.zip", flush=True)
        return 130
    finally:
        eval_env.close()
        env.close()

    if completed and model is not None:
        final_path = checkpoint_dir / "ppo_barrel_ryu_final"
        model.save(final_path)
        print(f"Final barrel model saved to: {final_path}.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
