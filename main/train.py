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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


BASE_LEVEL_ENV_COUNTS = (64,) * 12
DEFAULT_ENV_MULTIPLIER = 1
LEVEL_ENV_COUNTS = tuple(count * DEFAULT_ENV_MULTIPLIER for count in BASE_LEVEL_ENV_COUNTS)
LEVEL_COUNT = 12
NUM_ENVS = sum(LEVEL_ENV_COUNTS)
FEATURES_DIM = 1024
DEFAULT_N_STEPS = 128
DEFAULT_BATCH_SIZE = 8192
DEFAULT_N_EPOCHS = 3
OBSERVATION_MODE = "motion_rgb"
TIME_PENALTY = 0.0005
DEFAULT_LOG_DIR = "logs/ALL_MOTION_SIMULTANEOUS"


def build_worker_levels(
    level_env_counts: Sequence[int] = LEVEL_ENV_COUNTS,
) -> tuple[int, ...]:
    if len(level_env_counts) != LEVEL_COUNT:
        raise ValueError(f"level_env_counts must contain {LEVEL_COUNT} entries")
    if any(count <= 0 for count in level_env_counts):
        raise ValueError("every level must have at least one environment")
    return tuple(
        level
        for level, count in enumerate(level_env_counts)
        for _ in range(count)
    )


def scaled_level_env_counts(multiplier: int) -> tuple[int, ...]:
    if multiplier <= 0:
        raise ValueError("--env-multiplier must be positive")
    return tuple(count * multiplier for count in BASE_LEVEL_ENV_COUNTS)


def detect_device(requested: str) -> str:
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
    parser.add_argument("--total-timesteps", type=int, default=700_000_000)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--features-dim", type=int, default=FEATURES_DIM)
    parser.add_argument("--time-penalty", type=float, default=TIME_PENALTY)
    parser.add_argument("--checkpoint-timesteps", type=int, default=20_000_000)
    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument(
        "--env-multiplier",
        type=int,
        default=DEFAULT_ENV_MULTIPLIER,
        help="Scale the 64-worker-per-level profile when benchmarking larger runs.",
    )
    allocation.add_argument(
        "--envs-per-level", type=int,
        help="Override worker count per level (use 1 for a smoke run).",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="CPU threads for PPO; avoids oversubscription with emulator workers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Compatible PPO checkpoint to continue with the same 12-level mix.",
    )
    return parser


def validate_args(
    args: argparse.Namespace, level_env_counts: Sequence[int]
) -> None:
    positive_values = {
        "total-timesteps": args.total_timesteps,
        "n-steps": args.n_steps,
        "batch-size": args.batch_size,
        "n-epochs": args.n_epochs,
        "learning-rate": args.learning_rate,
        "features-dim": args.features_dim,
        "checkpoint-timesteps": args.checkpoint_timesteps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must exceed 1")
    if args.time_penalty < 0:
        raise ValueError("--time-penalty must be non-negative")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    if args.resume_model and not args.resume_model.is_file():
        raise FileNotFoundError(f"Resume model not found: {args.resume_model}")

    rollout_size = args.n_steps * sum(level_env_counts)
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch-size cannot exceed rollout size {rollout_size}"
        )
    if rollout_size % args.batch_size:
        raise ValueError(
            f"--batch-size must divide rollout size {rollout_size} exactly"
        )


def ppo_hyperparams(args: argparse.Namespace) -> dict[str, object]:
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


def write_training_config(
    log_dir: Path,
    args: argparse.Namespace,
    worker_levels: Sequence[int],
    level_env_counts: Sequence[int],
    device: str,
) -> None:
    config = {
        **vars(args),
        "resume_model": str(args.resume_model) if args.resume_model else None,
        "device_resolved": device,
        "level_env_counts": list(level_env_counts),
        "worker_levels": list(worker_levels),
        "num_envs": len(worker_levels),
        "observation_mode": OBSERVATION_MODE,
        "observation_shape_hwc": [84, 84, 3],
        "crop_size": [256, 200],
        "temporal_frames": [-16, -8, 0],
        "static_gain": 0.35,
    }
    (log_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    from main.env_utils import make_sfighter_env, validate_rom

    args = build_parser().parse_args(argv)
    level_env_counts = (
        (args.envs_per_level,) * LEVEL_COUNT
        if args.envs_per_level is not None
        else scaled_level_env_counts(args.env_multiplier)
    )
    build_worker_levels(level_env_counts)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    validate_args(args, level_env_counts)
    device = detect_device(args.device)
    validate_rom()

    worker_levels = build_worker_levels(level_env_counts)
    log_dir = Path(args.log_dir)
    checkpoint_dir = log_dir / "TRAINING"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_training_config(log_dir, args, worker_levels, level_env_counts, device)

    env = SubprocVecEnv(
        [
            make_sfighter_env(
                seed=args.seed + worker,
                obs_size=84,
                use_monitor=True,
                render=False,
                initial_level=level,
                time_penalty=args.time_penalty,
            )
            for worker, level in enumerate(worker_levels)
        ]
    )

    model: PPO | None = None
    completed = False
    try:
        hyperparams = ppo_hyperparams(args)
        if args.resume_model:
            print(f"Continuing model: {args.resume_model}", flush=True)
            model = PPO.load(
                str(args.resume_model),
                env=env,
                device=device,
                tensorboard_log=str(log_dir),
                custom_objects=hyperparams,
                print_system_info=False,
            )
            loaded_features_dim = model.policy.features_extractor.features_dim
            if loaded_features_dim != args.features_dim:
                raise ValueError(
                    "Resume model NatureCNN width mismatch: "
                    f"expected {args.features_dim}, got {loaded_features_dim}"
                )
            reset_num_timesteps = False
        else:
            model = PPO(
                "CnnPolicy",
                env,
                device=device,
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=args.seed,
                policy_kwargs={
                    "features_extractor_kwargs": {
                        "features_dim": args.features_dim,
                    }
                },
                **hyperparams,
            )
            reset_num_timesteps = True

        parameter_count = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        checkpoint_callback = CheckpointCallback(
            # CheckpointCallback counts vector steps.
            save_freq=max(args.checkpoint_timesteps // len(worker_levels), 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_motion_simultaneous_ryu",
        )
        print(
            f"Training one {args.features_dim}-feature NatureCNN "
            f"({parameter_count:,} policy parameters) on {device}",
            flush=True,
        )
        print(
            f"Environments={len(worker_levels)}, rollout={len(worker_levels) * args.n_steps:,}, "
            f"level counts={level_env_counts}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="PPO_Motion_Simultaneous_All",
            reset_num_timesteps=reset_num_timesteps,
        )
        completed = True
    except KeyboardInterrupt:
        if model is not None:
            interrupted_path = (
                checkpoint_dir / "ppo_motion_simultaneous_ryu_interrupted"
            )
            model.save(interrupted_path)
            print(f"Interrupted model saved to: {interrupted_path}.zip", flush=True)
        return 130
    finally:
        env.close()

    if completed and model is not None:
        final_path = checkpoint_dir / "ppo_sf2_ryu_motion_simultaneous_final"
        model.save(final_path)
        print(f"Final model saved to: {final_path}.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
