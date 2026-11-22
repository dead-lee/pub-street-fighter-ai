"""用和对战相同的 PPO、画面和网络，只在滚木桶上训练。

出招锁成轻拳升龙。模型要先对准桶再出手，而不是按固定节拍左右轮流打。
奖励是游戏实际加上的分数。升龙打空要扣分，桶不在这一侧还出招扣得更多。
等人站稳、以及站在落点下等桶下来，都不扣分。
对战默认 12 关各 64 个环境。木桶只有这一关，默认 8 个并行模拟器，把核留给网络更新。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import stable_baselines3.common.vec_env.subproc_vec_env as _subproc_vec_env

# 停训练时主进程先退出的话，子进程还在送这一步的结果，管道已经断了。
_SHUTDOWN = {"started": False}
_STOCK_WORKER = _subproc_vec_env._worker


def _quiet_worker(remote, parent_remote, env_fn_wrapper):
    """子进程退出时先关掉模拟器。管道断开不再把 Broken pipe 打进日志。"""
    opened = []
    original = env_fn_wrapper.var

    def tracking_fn():
        env = original()
        opened.append(env)
        return env

    env_fn_wrapper.var = tracking_fn
    try:
        _STOCK_WORKER(remote, parent_remote, env_fn_wrapper)
    except (BrokenPipeError, EOFError, ConnectionResetError, KeyboardInterrupt):
        pass
    finally:
        for env in opened:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except Exception:
            pass


def _request_stop(signum, frame):
    """SIGTERM 默认直接杀进程，来不及关掉子模拟器。转成键盘中断，走原来的保存和关闭。"""
    if _SHUTDOWN["started"]:
        return
    _SHUTDOWN["started"] = True
    raise KeyboardInterrupt


def _install_stop_signal() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def _close_running_env(env) -> None:
    """关掉环境。并行模拟器如果先死等这一步的结果，子进程还在算、主进程在等，两边都会停住。"""
    if env is None:
        return
    processes = getattr(env, "processes", None)
    remotes = getattr(env, "remotes", None)
    if not processes or not remotes or getattr(env, "closed", False):
        try:
            env.close()
        except Exception:
            pass
        return

    def drain() -> None:
        for remote in remotes:
            try:
                while remote.poll():
                    remote.recv()
            except Exception:
                pass

    drain()
    for remote in remotes:
        try:
            remote.send(("close", None))
        except Exception:
            pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and any(process.is_alive() for process in processes):
        drain()
        for process in processes:
            process.join(timeout=0.05)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
    for remote in remotes:
        try:
            remote.close()
        except Exception:
            pass
    env.waiting = False
    env.closed = True


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
    # 已经收成单一招的策略，PPO 采不到等待。采样时给每个动作留一个下限，确定性出手不变。
    parser.add_argument("--explore-eps", type=float, default=0.0)
    # 还没清完的那一份开局占多数环境。剩下的环境轮流打另外几份，避免把已经清完的忘掉。
    parser.add_argument("--focus-state", type=int, default=None)
    # 其余开局一个环境都不留。用来单独把某一份练到清完，再和另外两份合回去。
    parser.add_argument("--focus-only", action="store_true")
    # 只训练最后一层动作权重。画面特征不动，已经清完的开局不容易被带崩。
    parser.add_argument("--freeze-features", action="store_true")
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
    parser.add_argument(
        "--live-only", action="store_true",
        help="只在实战截下来的木桶开局上训练和评估，不用原来那一份存档。",
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


def barrel_index_for_worker(
    worker: int, num_envs: int, focus: int | None, n_states: int = 3, focus_only: bool = False,
) -> int | None:
    """多数环境打还没清完的那份，每种其他开局留一个环境继续巩固。"""
    if focus is None:
        return None
    if focus_only:
        return focus
    others = [index for index in range(n_states) if index != focus]
    reserved = min(len(others), max(num_envs - 1, 0))
    if worker >= num_envs - reserved:
        return others[worker - (num_envs - reserved)]
    return focus


def freeze_visual_features(model: PPO) -> None:
    """画面网络和价值网络保持原样，只让最后一层动作权重继续学。"""
    trainable = []
    for name, param in model.policy.named_parameters():
        param.requires_grad = name.startswith("action_net")
        if param.requires_grad:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("action head not found; visual features were not frozen")
    model.policy.optimizer = torch.optim.Adam(trainable, lr=model.learning_rate)


def soften_sampling(model: PPO, epsilon: float) -> None:
    """采样概率和原策略混合。每个动作至少有 epsilon/动作数 的概率。

    只改采样和更新时用的概率，不改网络本身。确定性评估仍走原来概率最高的那一招。
    """
    if epsilon <= 0.0:
        return
    if epsilon >= 1.0:
        raise ValueError("explore-eps must be < 1")
    policy = model.policy
    action_net = policy.action_net
    action_dist = policy.action_dist
    # 放在列表里，评估看到高分后可以把随机比例降下来。
    epsilon_box = [epsilon]
    model._explore_eps = epsilon_box

    def mixed(latent_pi):
        eps = epsilon_box[0]
        logits = action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        floor = eps / probs.shape[-1]
        mixed_probs = (1.0 - eps) * probs + floor
        return action_dist.proba_distribution(action_logits=torch.log(mixed_probs))

    policy._get_action_dist_from_latent = mixed


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
    if not 0.0 <= args.explore_eps < 1.0:
        raise ValueError("--explore-eps must be in [0, 1)")
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
        # 单份存档加到五万以上，才算这一关的桶清完、时间奖励也拿到了。
        self.best_clears = -1

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_every != 0:
            return True
        bonus = self.eval_env.env
        paths = getattr(bonus, "_barrel_paths", None)
        if not paths:
            bonus.reset()
            paths = bonus._barrel_paths
        # 有多份实战开局时，每一份都打一局，避免只抽到原来那份高分。
        if len(paths) > 1:
            scores = []
            names = []
            for index, path in enumerate(paths):
                bonus.force_barrel_index = index
                scores.append(play_barrel_episode(self.model, self.eval_env))
                names.append(os.path.basename(path))
            bonus.force_barrel_index = None
            print(
                f"Barrel eval at {self.num_timesteps:,}: "
                + " ".join(f"{name}={score}" for name, score in zip(names, scores)),
                flush=True,
            )
        else:
            scores = [
                play_barrel_episode(self.model, self.eval_env)
                for _ in range(self.eval_episodes)
            ]
            print(
                f"Barrel eval at {self.num_timesteps:,}: "
                f"scores={scores} mean={sum(scores) / len(scores):.0f}",
                flush=True,
            )
        mean_score = sum(scores) / len(scores)
        clears = sum(1 for score in scores if score >= 50000)
        # 一开始要随机，才不会停在只打碎几只桶的打法上。有一份已经上了两万五，再收紧。
        if max(scores) >= 25000 and float(self.model.ent_coef) > 0.001:
            self.model.ent_coef = 0.001
            epsilon_box = getattr(self.model, "_explore_eps", None)
            if epsilon_box is not None:
                epsilon_box[0] = min(epsilon_box[0], 0.02)
            print(
                f"Annealed barrel exploration at {self.num_timesteps:,}: "
                f"best layout {max(scores)}",
                flush=True,
            )
        if clears > self.best_clears or (
            clears == self.best_clears and mean_score > self.best_score
        ):
            self.best_clears = clears
            self.best_score = mean_score
            self.model.save(self.best_path)
            print(
                f"New best barrel model: {self.best_path}.zip "
                f"({mean_score:.0f}, clears={clears})",
                flush=True,
            )
        return True


def main(argv: Sequence[str] | None = None) -> int:
    from main.env_utils import make_barrel_env, validate_rom

    _install_stop_signal()
    # python -m 时当前模块名是 __main__。子进程只能按 main.train_barrel 这个名字导入，
    # 所以这里用导入到的那一份函数，两边才是同一个对象。
    import main.train_barrel as barrel_mod

    _subproc_vec_env._worker = barrel_mod._quiet_worker
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
            live_only=args.live_only,
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

    from main.custom_wrappers import (
        BARREL_APPROACH_SCALE,
        BARREL_KNOCKDOWN_PENALTY,
        BARREL_SCORE_SCALE,
        BARREL_SHAPING_GAMMA,
        BARREL_UNAIMED_PENALTY,
        BARREL_WHIFF_PENALTY,
    )

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
            f"- whiff * {BARREL_WHIFF_PENALTY:g} - wrong_move - blocked"
            f" - unaimed * {BARREL_UNAIMED_PENALTY:g}"
            f" - knockdown * {BARREL_KNOCKDOWN_PENALTY:g}"
            f" + approach γ={BARREL_SHAPING_GAMMA:g} scale={BARREL_APPROACH_SCALE:g}"
        ),
    }
    (log_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    focus_plan = [
        barrel_index_for_worker(
            worker, args.num_envs, args.focus_state, focus_only=args.focus_only,
        )
        for worker in range(args.num_envs)
    ]
    if args.focus_state is not None:
        print(f"Barrel env focus: {focus_plan}", flush=True)
    env = SubprocVecEnv(
        [
            make_barrel_env(
                seed=args.seed + worker,
                use_monitor=True,
                time_penalty=args.time_penalty,
                live_only=args.live_only,
                barrel_index=focus_plan[worker],
            )
            for worker in range(args.num_envs)
        ]
    )
    eval_env = make_barrel_env(
        seed=args.seed + 10_000,
        time_penalty=args.time_penalty,
        live_only=args.live_only,
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
        soften_sampling(model, args.explore_eps)
        if args.freeze_features:
            freeze_visual_features(model)
            print("Training the action head only.", flush=True)
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
            f"features={args.features_dim}, explore_eps={args.explore_eps:g}",
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
        # 关闭过程中再来一次停止信号，不再往外抛，否则子进程又会被管道断开。
        _SHUTDOWN["started"] = True
        for closing in (locals().get("eval_env"), locals().get("env")):
            _close_running_env(closing)

    if completed and model is not None:
        final_path = checkpoint_dir / "ppo_barrel_ryu_final"
        model.save(final_path)
        print(f"Final barrel model saved to: {final_path}.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
