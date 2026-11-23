"""ROM 检查和单局环境的组装。

训练会给 12 关各开一批子进程，每个子进程调用这里返回的工厂函数。
ROM 必须先在父进程里导入，子进程才能找到同一份游戏。
"""

import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor

from main.custom_wrappers import (
    BARREL_STATE_DIM,
    BARREL_STATE_PATH,
    STATE_PATHS,
    BarrelActionWrapper,
    BarrelBonusWrapper,
    SFighterActionWrapper,
    SFighterCustomWrapper,
)

# Stable-Retro 里这版《街头霸王 2：冠军特别版》美版的游戏编号。
GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 合法取得的 ROM 可以放在这里，也可以用环境变量 SF2_ROM_PATH 指到别处。
PROJECT_ROM_PATH = PROJECT_ROOT / "data" / "street-fighter-ii-special-champion-edition-usa.md"


def validate_rom() -> None:
    """确认当前 Python 环境里已经有这份 ROM，没有就从项目目录或环境变量导入。"""
    configured = os.environ.get("SF2_ROM_PATH")
    if configured:
        rom = Path(configured).expanduser().resolve()
        if not rom.is_file():
            raise FileNotFoundError(f"SF2_ROM_PATH does not point to a file: {rom}")
        retro.data.merge(str(rom), quiet=True)
    try:
        rom_path = retro.data.get_romfile_path(GAME_ID)
    except FileNotFoundError:
        if PROJECT_ROM_PATH.is_file():
            retro.data.merge(str(PROJECT_ROM_PATH), quiet=True)
        try:
            rom_path = retro.data.get_romfile_path(GAME_ID)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing ROM for {GAME_ID}. Set SF2_ROM_PATH to a legally owned "
                f"ROM, or place it at {PROJECT_ROM_PATH}."
            ) from exc
    # 环境变量和已经导入的 ROM 必须是同一个文件，避免悄悄换了一版游戏。
    if configured and Path(rom_path).read_bytes() != rom.read_bytes():
        raise ValueError("SF2_ROM_PATH is not the supported Street Fighter II ROM")
    print(f"Using ROM: {rom_path}", flush=True)


def make_sfighter_env(seed=0, obs_size=84, use_monitor=False, render=False,
                      initial_level=0, time_penalty=0.0005, render_skip=1,
                      reset_round=False, auto_reset=False, max_episode_steps=18000,
                      realtime=False):
    """返回一个子进程能调用的环境工厂。ROM 要在父进程里先导入。

    realtime 只给观看用：全屏、原速、有声音。训练必须保持 False，
    否则每个模拟器都会去开窗口、抢声卡。
    """
    if not isinstance(initial_level, int) or not 0 <= initial_level < len(STATE_PATHS):
        raise ValueError("initial_level must be an integer between 0 and 11")
    if obs_size < 36 or max_episode_steps < 1 or time_penalty < 0:
        raise ValueError("obs_size must be >= 36, max_episode_steps positive, time_penalty non-negative")
    state_path = Path(STATE_PATHS[initial_level])
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing level save-state: {state_path}")

    def _init():
        # 每个模拟器进程只留一个 OpenCV 线程，避免和并行环境抢 CPU。
        cv2.setNumThreads(1)
        env = retro.make(
            game=GAME_ID,
            # 不使用集成自带的默认存档，开局存档由包装器按关卡加载。
            state=retro.State.NONE,
            obs_type=retro.Observations.IMAGE,
            # 画面自己画。human 模式会再开一个 pyglet 小窗。
            render_mode="rgb_array",
            info=str(PROJECT_ROOT / "data" / "data.json"),
            scenario=str(PROJECT_ROOT / "data" / "scenario_full_frame.json"),
        )
        try:
            env = SFighterCustomWrapper(
                env, obs_size=obs_size, render=render, initial_level=initial_level,
                time_penalty=time_penalty, render_skip=render_skip, reset_round=reset_round,
                max_episode_steps=max_episode_steps, realtime=realtime,
            )
            # 把 27 个离散动作展开成手柄按键，顺序必须和已保存的模型一致。
            env = SFighterActionWrapper(env)
            if use_monitor:
                env = Monitor(env)
            env.action_space.seed(seed)
            if auto_reset:
                env.reset(seed=seed)
            return env
        except BaseException:
            env.close()
            raise

    return _init


def make_barrel_env(seed=0, obs_size=84, use_monitor=False, time_penalty=0.0005,
                    max_episode_steps=900, render=False, realtime=False,
                    live_only=False, barrel_index=None, state_obs=False):
    """木桶训练用的环境工厂。

    state_obs 时，观测是桶的位置和速度，出招含左右波动拳。
    否则仍是和对战相同的运动图，旧的五动作模型还能直接打。
    一局大约 550 个动作就会切去下一场。900 是超时上限，正常在切场时结束。
    live_only 时不加载原来那份木桶存档，只练实战里截下来的开局。
    barrel_index 给定时，这个环境每次都从那一份开局打起。
    """
    if obs_size < 36 or max_episode_steps < 1 or time_penalty < 0:
        raise ValueError("obs_size must be >= 36, max_episode_steps positive, time_penalty non-negative")
    state_path = Path(BARREL_STATE_PATH)
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing barrel save-state: {state_path}")

    def _init():
        cv2.setNumThreads(1)
        env = retro.make(
            game=GAME_ID,
            state=retro.State.NONE,
            obs_type=retro.Observations.IMAGE,
            render_mode="rgb_array",
            info=str(PROJECT_ROOT / "data" / "data.json"),
            scenario=str(PROJECT_ROOT / "data" / "scenario_full_frame.json"),
        )
        try:
            env = BarrelBonusWrapper(
                env, obs_size=obs_size, render=render, initial_level=0,
                time_penalty=time_penalty, max_episode_steps=max_episode_steps,
                realtime=realtime,
            )
            env.live_only = live_only
            env.state_obs = bool(state_obs)
            if env.state_obs:
                # 位置向量的每一维都压进这个范围，网络不用再看画面。
                env.observation_space = gym.spaces.Box(
                    low=-3.0, high=3.0, shape=(BARREL_STATE_DIM,), dtype=np.float32,
                )
            # 指定某一份开局时，这个环境每次都打它，用来把梯度集中到还没清完的那份。
            if barrel_index is not None:
                env.force_barrel_index = int(barrel_index)
            env = BarrelActionWrapper(env)
            if use_monitor:
                env = Monitor(env)
            env.action_space.seed(seed)
            return env
        except BaseException:
            env.close()
            raise

    return _init
