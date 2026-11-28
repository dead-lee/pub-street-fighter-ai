# Copyright 2026 Lin Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""对战的画面、血量奖励，以及 27 个动作。

连续闯关走到木桶时，再交给 barrel_play。
"""

import collections
import os

import cv2
import gymnasium as gym
import numpy as np

from main.bonus import BonusRoutine
from main.buttons import NO_OP_ACTION
from main.saves import STATE_PATHS
from main.score_skip import RESULT_SKIP_SCORE_JUMP, result_skip_action
from main.stages import BONUS_BARREL, BONUS_STAGE_NAMES

# 快速预览时每画一帧只等 1 毫秒，所以默认比游戏原速快很多。
FRAME_TIME = 1  # 毫秒。
# 等下一回合或下一场对手时的上限。3600 帧大约 60 秒。
# 连续闯关的世界地图过场更长，调用方会另外传入更大的上限。
GAME_END_WAIT_STEPS = 3600  # 模拟器帧数。

class SFighterCustomWrapper(gym.Wrapper):
    """把一局街机对战变成强化学习能用的观测、奖励和结束信号。"""

    def __init__(self, env, obs_size=84, render=False, initial_level=0,
                 time_penalty=0.0005,
                 render_skip=1, reset_round=False, max_episode_steps=18000,
                 realtime=False, record_path=None):
        super(SFighterCustomWrapper, self).__init__(env)

        self.states = [os.path.abspath(fn) for fn in STATE_PATHS]
        self.level_id = initial_level

        self.time_penalty = float(time_penalty)
        # 模型输入由 5 帧处理后的画面拼出来，见 _stack_observation。
        self.stack_size = 5
        self.frame_stack = collections.deque(maxlen=self.stack_size)

        # 这版游戏满血是 176。内存里血量变成负数表示已经被击倒。
        self.full_hp = 176.0
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        self.obs_size = obs_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8)

        self.agent_wins = 0
        self.enemy_wins = 0
        # 一回合分出胜负后，先空转等下一回合开打，这段不交给模型决策。
        self.waiting_for_next_round = False
        self._bonus = BonusRoutine()

        # 打掉对手的血比自己掉血更值钱，系数是 3。
        self.attk_reward_coeff = 3.0

        self.render_enabled = render
        # 原速全屏只在确实要画出来时启用。训练关着画面，这里保持关闭。
        self.realtime = bool(realtime) and bool(render)
        self._record_path = record_path
        self._presenter = None
        self.render_skip = int(render_skip)
        if self.render_skip < 1:
            raise ValueError("render_skip must be at least 1")
        self._render_counter = 0
        # 打开后，一次击倒就结束这一局，不再打满三局两胜。
        self.reset_round = bool(reset_round)
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self._match_decided = False
        self._reset_result_skip()

    def _process_frame(self, frame):
        """裁掉画面边缘，把静止背景压暗，只把运动的人留亮，再缩到 84×84。"""
        height, width = frame.shape[:2]
        crop_width, crop_height = min(256, width), min(200, height)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        frame = frame[top:top + crop_height, left:left + crop_width]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._motion_frames.append(gray)
        if len(self._motion_frames) >= 5:
            # 下标 [-5, -3, -1] 对应模拟器时间 t-16、t-8、t。
            # 一个模型动作含 4 帧，所以隔两个样本就是隔 8 帧。
            first = cv2.absdiff(self._motion_frames[-5], self._motion_frames[-3])
            second = cv2.absdiff(self._motion_frames[-3], self._motion_frames[-1])
            motion = cv2.GaussianBlur(np.maximum(first, second), (5, 5), 0)
            _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            # 运动区域保留原色，背景亮度乘 0.35，让模型更容易看见人。
            frame = np.where(mask[..., None] > 0, frame, frame * 0.35).astype(np.uint8)
        return cv2.resize(frame, (self.obs_size, self.obs_size), interpolation=cv2.INTER_LINEAR)

    def _stack_observation(self):
        """用三张不同时刻的画面各取一个颜色通道，拼成模型要的 84×84×3。"""
        frames = list(self.frame_stack)
        return np.stack([frames[0][..., 0], frames[2][..., 1], frames[4][..., 2]], axis=-1)

    def _reset_result_skip(self):
        """一轮结算开始时清掉跳分状态。"""
        self._result_frames = 0
        self._result_prev_score = None
        self._tick_started_at = None
        self._result_stable = 0
        self._score_skip_done = False

    def _note_result_score(self, info):
        """记住分数从哪一帧开始跳，以及轻拳有没有把剩余分数一次加上。"""
        score = info.get("score")
        if score is None:
            return
        score = int(score)
        if self._result_prev_score is None:
            self._result_prev_score = score
            return
        delta = score - self._result_prev_score
        if delta >= RESULT_SKIP_SCORE_JUMP:
            self._score_skip_done = True
        elif delta > 0:
            if self._tick_started_at is None:
                self._tick_started_at = self._result_frames
            self._result_stable = 0
        elif self._tick_started_at is not None:
            self._result_stable += 1
            # 跳分已经停了。空隙留大一点，免得两次加分之间被当成结束。
            if self._result_stable >= 45:
                self._score_skip_done = True
        self._result_prev_score = score
        # 胜利姿势比跳分来得早，秒杀时姿势能亮到 400 帧左右。
        # 失败结算有时几乎不加分，姿势看过一会儿就放行，不要在跳分开始前就放弃。
        if self._tick_started_at is None and self._result_frames >= 900:
            self._score_skip_done = True

    def reset(self, **kwargs):
        """读入当前关卡的开局存档，并把前几帧观测填满。"""
        self.elapsed_steps = 0

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        self.agent_wins = 0
        self.enemy_wins = 0
        self.waiting_for_next_round = False
        self._match_decided = False
        self._reset_result_skip()
        self._motion_frames = collections.deque(maxlen=5)

        self.env.unwrapped.load_state(self.states[self.level_id])
        obs, info = self.env.reset(**kwargs)
        obs = self._process_frame(obs)
        self.frame_stack.clear()
        # 开局还没有历史帧，先用第一帧复制满整个栈。
        self.frame_stack.extend([obs] * self.stack_size)
        return self._stack_observation(), info

    def advance_to_next_match(self, max_steps=GAME_END_WAIT_STEPS):
        """不读下一关存档，让游戏自己播完过场，直到下一位对手开打。

        胜利台词和世界地图保持空按键。汽车和砖墙用固定套路。
        滚木桶用和三份存档同一套规则，不把这种关卡交给只见过对战画面的模型。
        双方血量回到 100 以上，而且场景仍是对战场地，才交还模型。
        """
        info = {}
        announced_bonus = set()
        steps = 0
        # max_steps 为 None 时一直跟着游戏走，直到下一场开打，或用户关掉窗口。
        while max_steps is None or steps < max_steps:
            steps += 1
            if self.render_enabled:
                self.render_game()
            stage = info.get("stage")
            stage_id = int(stage) if stage is not None else -1
            in_bonus = stage_id > 11
            if stage_id == BONUS_BARREL:
                if stage_id not in announced_bonus:
                    announced_bonus.add(stage_id)
                    print("Bonus stage: 滚木桶. Using the barrel rule.", flush=True)
                # 一整关都由木桶模型打完再回到这里。中途不能把观测栈清掉重来。
                info = self._play_barrel_stage()
                continue
            if in_bonus:
                if stage_id not in announced_bonus:
                    announced_bonus.add(stage_id)
                    name = BONUS_STAGE_NAMES.get(stage_id, str(stage_id))
                    print(f"Bonus stage: {name}. Using a fixed routine.", flush=True)
                # 汽车和砖墙看的是模拟器原始画面，不是模型用的那张缩小图。
                screen = self.env.render()
                action = self._bonus.action(stage_id, info, screen)
            else:
                action = NO_OP_ACTION
            obs, _, _, _, info = self.env.step(action)
            # 场景配置会在回合重置计数到 10 时标记结束。过场和奖励关也会碰到这个计数，
            # 游戏本身还在往下走，不能据此把连续闯关掐掉。
            stage = info.get("stage")
            fight_ready = info['agent_hp'] > 100 and info['enemy_hp'] > 100
            # 场景编号 0–11 是 12 个对战场地。bonus 没有正常的双方满血，
            # 不能在连打过程中把控制权交还给模型。
            if stage is not None and int(stage) > 11:
                fight_ready = False
            if fight_ready:
                self.prev_player_health = info['agent_hp']
                self.prev_opponent_health = info['enemy_hp']
                self.elapsed_steps = 0
                self.agent_wins = 0
                self.enemy_wins = 0
                self.waiting_for_next_round = False
                self._motion_frames = collections.deque(maxlen=5)
                obs = self._process_frame(obs)
                self.frame_stack.clear()
                self.frame_stack.extend([obs] * self.stack_size)
                return self._stack_observation(), info
        raise RuntimeError(
            f"Next match did not start within {max_steps} emulator steps"
        )

    def _play_barrel_stage(self):
        """用三份存档上打通的那套规则打完这一关。不读木桶存档，就在当前画面上打。

        飞入动画时关卡号已经是木桶，人还在画面外。等人站稳再出手。
        画面和声音仍走这场闯关的窗口，不另开一个。
        """
        from main.barrel_play import BARREL_GROUND_Y, BarrelBonusWrapper, barrel_action_sequences
        for _ in range(600):
            info = self.env.unwrapped.data.lookup_all()
            if int(info.get("stage", BONUS_BARREL)) != BONUS_BARREL:
                return info
            if int(info.get("agent_x", 0)) > 0 and int(info.get("agent_y", 0)) >= BARREL_GROUND_Y:
                break
            if self.render_enabled:
                self.render_game()
            self.env.step(NO_OP_ACTION)
        player = getattr(self, "_barrel_rule", None)
        if player is None:
            player = BarrelBonusWrapper(
                self.env,
                render=self.render_enabled,
                realtime=self.realtime,
                max_episode_steps=20000,
            )
            self._barrel_rule = player
        # 出招循环走规则自己的 step。画图仍用这场闯关的窗口，避免再开一块屏幕、再占一次声卡。
        player.render_enabled = self.render_enabled
        player.render_game = self.render_game
        player.begin_live()
        info = self.env.unwrapped.data.lookup_all()
        done = False
        truncated = False
        while not done and not truncated:
            frames = barrel_action_sequences()[int(player.rule_action())]
            _obs, _reward, done, truncated, info = player.step(frames)
        gained = int(info.get("barrel_gained", 0))
        print(f"Barrel rule score: +{gained}", flush=True)
        stage = int(info.get("stage", BONUS_BARREL))
        idle_frames = 0
        while stage == BONUS_BARREL and idle_frames < 6000:
            idle_frames += 1
            if self.render_enabled:
                self.render_game()
            _obs, _reward, _terminated, _truncated, info = self.env.step(NO_OP_ACTION)
            stage = int(info.get("stage", BONUS_BARREL))
        return info

    def step(self, action):
        """执行一个模型动作，并按血量变化给出奖励。

        action 已经是 4 帧手柄按键。一帧模拟器只吃一组键，
        所以这里把这 4 组键依次送进去，模型每决策一次，游戏前进 4 帧。
        """
        self.elapsed_steps += 1
        custom_done = False
        custom_truncated = False
        action_advanced = False

        wait_steps = 0
        # 上一回合的胜负画面：先亮 1 秒，再点按把跳分跳过，然后才进下一回合。
        while self.waiting_for_next_round:
            wait_steps += 1
            if wait_steps > GAME_END_WAIT_STEPS:
                info["truncation_reason"] = "next_round_timeout"
                custom_truncated = True
                break
            if self.render_enabled:
                self.render_game()
            # 分数开始跳之后再等 1 秒，点一次轻拳把剩下的跳分跳掉。
            self._result_frames += 1
            since_tick = None
            if self._tick_started_at is not None:
                since_tick = self._result_frames - self._tick_started_at
            skip_action = result_skip_action(since_tick, self._score_skip_done)
            obs, _, _, truncated, info = self.env.step(skip_action)
            if truncated:
                custom_truncated = True
                break
            self._note_result_score(info)
            curr_player_health = info['agent_hp']
            curr_opponent_health = info['enemy_hp']

            # 整场已经分出胜负。结算一跳完就结束这一局，后面的地图交给过场。
            if self._match_decided and self._score_skip_done:
                self.waiting_for_next_round = False
                obs = self._process_frame(obs)
                self.frame_stack.append(obs)
                return self._stack_observation(), 0.0, True, False, info

            # 两边血量都回到 100 以上，说明下一回合已经开始。
            if curr_player_health > 100 and curr_opponent_health > 100:

                self.waiting_for_next_round = False
                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health

                obs = self._process_frame(obs)
                self.frame_stack.append(obs)
                truncated = self.elapsed_steps >= self.max_episode_steps
                if truncated:
                    info["truncation_reason"] = "match_step_limit"
                return self._stack_observation(), 0.0, False, truncated, info

        if not custom_truncated:
            for frame_action in action:
                if self.render_enabled:
                    self.render_game()
                obs, _, _, truncated, info = self.env.step(frame_action)
                action_advanced = True
                if truncated:
                    custom_truncated = True
                    break

        obs = self._process_frame(obs)
        self.frame_stack.append(obs)

        curr_player_health = info['agent_hp']
        curr_opponent_health = info['enemy_hp']

        # 奖励 = 0.01 × (3 × 对手掉血 − 自己掉血)，击倒另有加减。
        damage_dealt = self.prev_opponent_health - curr_opponent_health
        damage_taken = self.prev_player_health - curr_player_health
        custom_reward = self.attk_reward_coeff * damage_dealt - damage_taken

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # 同归于尽、被击倒、击倒对手，都先停在胜负画面上。
        # 分数会在这张画面里慢慢往上加，等 1 秒后用轻拳跳过。
        # reset_round 打开时不看结算，一次击倒就结束。
        round_over = (
            curr_player_health < 0 or curr_opponent_health < 0
        )
        if curr_player_health < 0 and curr_opponent_health >= 0:
            custom_reward -= (self.full_hp + curr_opponent_health)
            self.enemy_wins += 1
        elif curr_opponent_health < 0 and curr_player_health >= 0:
            custom_reward += (self.full_hp + curr_player_health)
            self.agent_wins += 1

        if round_over and not self.reset_round:
            self.waiting_for_next_round = True
            self._reset_result_skip()
            # 三局两胜已经有人拿到两回合，这场结束，但先把结算画面播完。
            if self.agent_wins >= 2 or self.enemy_wins >= 2:
                self._match_decided = True
        elif round_over and self.reset_round:
            custom_done = True
        reward = 0.01 * custom_reward
        if action_advanced:
            reward -= self.time_penalty
        if self.elapsed_steps >= self.max_episode_steps and not custom_done:
            custom_truncated = True
            info["truncation_reason"] = "match_step_limit"
        return self._stack_observation(), reward, custom_done, custom_truncated, info

    def close(self):
        try:
            self.env.close()
        finally:
            if self._presenter is not None:
                self._presenter.close()
                self._presenter = None
            elif self.render_enabled:
                cv2.destroyAllWindows()

    def render_game(self):
        """原速模式每帧都画并出声；快速预览则跳帧，而且几乎不等待。"""
        if self.realtime:
            self._present_realtime()
            return
        self._render_counter += 1
        if self._render_counter % self.render_skip != 0:
            return
        frame = self.env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Street Fighter II', frame)
        cv2.waitKey(FRAME_TIME)

    def _present_realtime(self):
        """第一次画的时候才开全屏窗口，避免环境刚建好就占住屏幕。"""
        if self._presenter is None:
            from main.presenter import RealtimePresenter
            emulator = self.env.unwrapped
            self._presenter = RealtimePresenter(
                sample_rate=float(emulator.em.get_audio_rate()),
                frame_rate=float(emulator.em.get_screen_rate()),
                record_path=self._record_path,
            )
        frame = self.env.render()
        # get_audio() 取走的是刚刚这一帧模拟器产生的立体声。
        audio = self.env.unwrapped.em.get_audio()
        self._presenter.present(frame, audio)

class SFighterActionWrapper(gym.ActionWrapper):
    """把模型的 0–26 号动作变成 4 帧 MD 手柄按键。顺序不能改，否则旧模型对不上。"""

    def __init__(self, env):
        super().__init__(env)
        # Stable-Retro 里这颗 MD 手柄的按键顺序。
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
                   'C', 'Y', 'X', 'Z')

        def frame(*pressed):
            return [int(button in pressed) for button in buttons]

        # 0：站立。1–4：四个方向。5–8：四个斜向。每个动作按住 4 帧。
        self._actions = [[frame()] * 4]
        for direction in ('LEFT', 'RIGHT', 'UP', 'DOWN'):
            self._actions.append([frame(direction)] * 4)
        for vertical, horizontal in (('DOWN', 'LEFT'), ('DOWN', 'RIGHT'),
                                     ('UP', 'LEFT'), ('UP', 'RIGHT')):
            self._actions.append([frame(vertical, horizontal)] * 4)
        # 9–14：站立时的六个攻击键。15–20：蹲着按同样的六个键。
        attacks = ('A', 'B', 'C', 'X', 'Y', 'Z')
        for attack in attacks:
            self._actions.append([frame(attack)] * 4)
        for attack in attacks:
            self._actions.append([frame('DOWN', attack)] * 4)

        # 21–23：面向右的波动拳、旋风腿、升龙拳。24–26：面向左，方向相反。
        # 每招正好拆成 4 帧，和上面的普通动作一样长。
        for forward, backward in (('RIGHT', 'LEFT'), ('LEFT', 'RIGHT')):
            self._actions.extend([
                [frame('DOWN'), frame('DOWN', forward), frame(forward),
                 frame(forward, 'Y')],
                [frame('DOWN'), frame('DOWN', backward), frame(backward),
                 frame(backward, 'B')],
                [frame(forward), frame('DOWN'), frame('DOWN', forward),
                 frame('DOWN', forward, 'Y')],
            ])
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]
