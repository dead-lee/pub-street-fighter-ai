import os
from pathlib import Path

import cv2
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor

from main.custom_wrappers import STATE_PATHS, SFighterActionWrapper, SFighterCustomWrapper

GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROM_PATH = PROJECT_ROOT / "data" / "street-fighter-ii-special-champion-edition-usa.md"


def validate_rom() -> None:
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
    if configured and Path(rom_path).read_bytes() != rom.read_bytes():
        raise ValueError("SF2_ROM_PATH is not the supported Street Fighter II ROM")
    print(f"Using ROM: {rom_path}", flush=True)


def make_sfighter_env(seed=0, obs_size=84, use_monitor=False, render=False,
                      initial_level=0, time_penalty=0.0005, render_skip=1,
                      reset_round=False, auto_reset=False, max_episode_steps=18000):
    """Create a worker factory; initialize the ROM in the parent first."""
    if not isinstance(initial_level, int) or not 0 <= initial_level < len(STATE_PATHS):
        raise ValueError("initial_level must be an integer between 0 and 11")
    if obs_size < 36 or max_episode_steps < 1 or time_penalty < 0:
        raise ValueError("obs_size must be >= 36, max_episode_steps positive, time_penalty non-negative")
    state_path = Path(STATE_PATHS[initial_level])
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing level save-state: {state_path}")

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
            env = SFighterCustomWrapper(
                env, obs_size=obs_size, render=render, initial_level=initial_level,
                time_penalty=time_penalty, render_skip=render_skip, reset_round=reset_round,
                max_episode_steps=max_episode_steps,
            )
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
