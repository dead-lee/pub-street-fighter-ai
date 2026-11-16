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

"""画面预处理、奖励、对战流程，以及 27 个动作到手柄按键的映射。

模型看到的不是原始游戏画面，而是裁剪、压暗背景后的 84×84 运动图。
奖励只根据双方血量变化计算。胜负按街机规则：三局两胜，血量读内存。
"""

import os
import collections

import cv2
import numpy as np
import gymnasium as gym

# 快速预览时每画一帧只等 1 毫秒，所以默认比游戏原速快很多。
FRAME_TIME = 1  # 毫秒。
# 等下一回合或下一场对手时的上限。3600 帧大约 60 秒。
# 连续闯关的世界地图过场更长，调用方会另外传入更大的上限。
GAME_END_WAIT_STEPS = 3600  # 模拟器帧数。

# 世嘉 MD 手柄 12 个键都松开。过场和回合间用这个空动作往前走。
NO_OP_ACTION = np.zeros(12, dtype=np.int8)

# 12 维按键的下标，顺序和 SFighterActionWrapper 里的 buttons 一致。
# X/Y/Z 是轻、中、重拳，A/B/C 是轻、中、重踢。
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7
BUTTON_C = 8   # 重踢。车的左边先踢开，右边上半截也用它；砖堆贴身后它够得比拳头远。
BUTTON_Y = 9   # 中拳。拆砖的主力，车的底盘也靠蹲着的中拳。收招快，一段里能多打几下。
BUTTON_X = 10  # 轻拳。结算画面上点一下跳过跳分；木桶用它出升龙，拳轻，收招比重拳短。
BUTTON_Z = 11  # 重拳。砸车时用来出升龙，这一下能把人带到车顶再落到右边。

# 分数开始一格格累加之后，先看 1 秒，再点一次轻拳把剩下的一次加上。
# 胜利姿势刚出来时按键不会跳过结算，所以要等分数真的开始跳。
RESULT_HOLD_FRAMES = 60
RESULT_TAP_FRAMES = 3
# 正常是一小截一小截地加。一次加上八千以上，就是跳过已经生效。
RESULT_SKIP_SCORE_JUMP = 8000

# data.json 里的 stage。0–11 是对战场地，这三关是街机插进来的 bonus。
BONUS_CAR = 12
BONUS_WALL = 13
BONUS_BARREL = 14
BONUS_STAGE_NAMES = {
    BONUS_CAR: "砸汽车",
    BONUS_WALL: "打砖墙",
    BONUS_BARREL: "滚木桶",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 12 关的开局存档，顺序就是街机进度：盖尔、肯、春丽……最后是拜森。
# 难度字节已经是菜单里的 8 星。逐关读档会跳过关与关之间的世界地图。
STATE_PATHS = [
    os.path.join(DATA_DIR, "Champion.X.Level01.RyuVsGuile.state"),
    os.path.join(DATA_DIR, "Champion.X.Level02.RyuVsKen.state"),
    os.path.join(DATA_DIR, "Champion.X.Level03.RyuVsChunLi.state"),
    os.path.join(DATA_DIR, "Champion.X.Level04.RyuVsZangief.state"),
    os.path.join(DATA_DIR, "Champion.X.Level05.RyuVsDhalsim.state"),
    os.path.join(DATA_DIR, "Champion.X.Level06.RyuVsRyu.state"),
    os.path.join(DATA_DIR, "Champion.X.Level07.RyuVsHonda.state"),
    os.path.join(DATA_DIR, "Champion.X.Level08.RyuVsBlanka.state"),
    os.path.join(DATA_DIR, "Champion.X.Level09.RyuVsBalrog.state"),
    os.path.join(DATA_DIR, "Champion.X.Level10.RyuVsVega.state"),
    os.path.join(DATA_DIR, "Champion.X.Level11.RyuVsSagat.state"),
    os.path.join(DATA_DIR, "Champion.X.Level12.RyuVsBison.state"),
]
# 滚木桶开打的存档。训练只从这里重开，不走前面的对战。
BARREL_STATE_PATH = os.path.join(DATA_DIR, "Champion.X.BonusBarrel.state")


def _blank_action():
    return np.zeros(12, dtype=np.int8)


def _barrels_near(prev, img, ryu_x):
    """上一帧和这一帧对不上的色块，就是正在掉或正在滚的桶。

    滑槽正前方、高度已经进拳头范围的，算右边能打的。
    贴着地面、人左侧不远处的，算要转身去打的。
    整行一起闪的是背景动画，不算桶。
    """
    if prev is None or img is None or prev.shape != img.shape:
        return False, False
    diff = np.abs(img.astype(np.int16) - prev.astype(np.int16)).sum(axis=2)
    mask = (diff > 110).astype(np.uint8)
    mask[:40] = 0
    mask[190:] = 0
    mask[mask.sum(axis=1) > 50] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    right_falling = False
    left_ground = False
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        if not (50 <= area <= 1800 and height >= 12 and width <= 64):
            continue
        x, y = centroids[i]
        dx = x - ryu_x
        if 0 <= dx <= 36 and 85 <= y <= 140:
            right_falling = True
        if 8 < (ryu_x - x) < 50 and y >= 150:
            left_ground = True
    return right_falling, left_ground


class BonusRoutine:
    """三关 bonus 的固定套路。不看模型，只看分数有没有还在涨、人站在哪。

    砸汽车：左边贴身一记重腿。龙和肯贴得很近时这一脚自己打两段，
    接着重拳升龙翻过车顶落到右边，上半截用重踢，底盘蹲着打。
    打砖墙：先拆左边，再走近用重踢拆中间，最后中拳收尾。
    滚木桶：照 Ken 的打法。轻拳升龙等桶落到拳头高度再发，
    桶滚到左边就转身打那一侧，人被升龙带过滑槽就走回来等下一只。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = None
        self.started = False
        self.frame = 0
        self.phase = 0
        self.phase_frame = 0
        self.quiet = 0
        self.saw_score = False
        self.prev_score = None
        self.prev_x = None
        self.still = 0
        self.crossed = False
        # 木桶：上一帧画面、升龙打到第几帧、下一发最早能出的帧、这一发朝哪边。
        self.barrel_prev = None
        self.barrel_age = None
        self.barrel_ready = 0
        self.barrel_forward = BUTTON_RIGHT

    def action(self, stage, info, screen=None):
        """返回这一帧的 12 键。人还没站到场上时什么都不按。

        screen 是当前原始画面。木桶要靠它看桶落在哪一侧。
        """
        stage = int(stage)
        if stage != self.stage:
            self.reset()
            self.stage = stage
        if not self.started:
            hp = int(info.get("agent_hp", 0))
            x = int(info.get("agent_x", 0))
            if hp > 100 and x > 0:
                self.started = True
                self.prev_score = int(info.get("score", 0))
            else:
                return _blank_action()
        frame = self.frame
        self.frame += 1
        if stage == BONUS_CAR:
            return self._car(frame, info)
        if stage == BONUS_WALL:
            return self._brick(frame, info)
        return self._barrel(frame, info, screen)

    def _note_score(self, info):
        score = int(info.get("score", 0))
        if self.prev_score is None:
            self.prev_score = score
            return
        if score != self.prev_score:
            self.prev_score = score
            self.saw_score = True
            self.quiet = 0
        else:
            self.quiet += 1

    def _begin_phase(self, phase):
        self.phase = phase
        self.phase_frame = 0
        self.quiet = 0
        self.saw_score = False
        self.still = 0

    def _mash(self, info, button, period, quiet, crouch, cap):
        """连打，直到这一段不再加分。按下 2 帧，松开到收招结束再按下一发。"""
        self._note_score(info)
        done = (
            (self.saw_score and self.quiet >= quiet)
            or self.phase_frame >= cap
        )
        action = _blank_action()
        if crouch:
            action[BUTTON_DOWN] = 1
        if self.phase_frame % period < 2:
            action[button] = 1
        self.phase_frame += 1
        if done:
            self._begin_phase(self.phase + 1)
        return action

    def _car(self, frame, info):
        # 进场这三百帧踢出去打不中车，升龙也出不来。
        if frame < 300:
            return _blank_action()
        if self.phase == 0:
            return self._car_kicks()
        if self.phase == 1:
            return self._car_shory()
        if self.phase == 2:
            return self._car_to_right(info)
        if self.phase == 3:
            # 落到右边以后，重踢能打到上半截。站着打，底盘还在。
            return self._mash(info, BUTTON_C, period=22, quiet=80, crouch=False, cap=1100)
        # 上半截烂掉以后站立踢够不着底盘，要蹲下去。
        return self._mash(info, BUTTON_Y, period=16, quiet=80, crouch=True, cap=1200)

    def _car_kicks(self):
        """贴身一记重腿。龙和肯贴着车时这一脚自己打两段，不用再补第二脚。

        第二段大约在踢出后 30 帧才加上。整脚收招要 50 多帧，
        这段里再按键，升龙出不来。
        """
        if self.phase_frame >= 56:
            self._begin_phase(1)
            return self._car_shory()
        t = self.phase_frame
        self.phase_frame += 1
        action = _blank_action()
        # 按下两帧，剩下的帧把两段伤害和收招让完。
        if t < 2:
            action[BUTTON_C] = 1
        return action

    def _car_shory(self):
        """重拳升龙。方向键按满两帧才会升起来，一帧只是往前挪一下。"""
        if self.phase_frame >= 58:
            self._begin_phase(2)
            return self._car_to_right_step()
        t = self.phase_frame
        self.phase_frame += 1
        action = _blank_action()
        # 8 帧出招，再留 50 帧人在空中，不要中途改方向。
        if t < 8:
            part = t // 2
            if part == 0:
                action[BUTTON_RIGHT] = 1
            elif part == 1:
                action[BUTTON_DOWN] = 1
            else:
                action[BUTTON_DOWN] = 1
                action[BUTTON_RIGHT] = 1
                if part == 3:
                    action[BUTTON_Z] = 1
        return action

    def _car_to_right(self, info):
        """升龙把人带到车顶上方。按住右走到车子外面落地，再转回来面对车。"""
        x = int(info.get("agent_x", 0))
        y = int(info.get("agent_y", 0))
        if not self.crossed:
            if (y >= 190 and x >= 300) or self.phase_frame > 180:
                self.crossed = True
                self.phase_frame = 0
            else:
                self.phase_frame += 1
                action = _blank_action()
                action[BUTTON_RIGHT] = 1
                return action
        if self.phase_frame >= 6:
            self.crossed = False
            self._begin_phase(3)
            return _blank_action()
        self.phase_frame += 1
        action = _blank_action()
        action[BUTTON_LEFT] = 1
        return action

    def _car_to_right_step(self):
        """升龙的最后一帧直接接上往右走，避免空一帧从车顶掉回去。"""
        self.phase_frame = 1
        action = _blank_action()
        action[BUTTON_RIGHT] = 1
        return action

    def _brick(self, frame, info):
        if frame < 30:
            return _blank_action()
        if self.phase == 0:
            # 砖堆的左下角。打穿以后人贴不进去，要往右迈一步。
            return self._mash(info, BUTTON_Y, period=14, quiet=70, crouch=False, cap=900)
        if self.phase == 1:
            return self._step_into_bricks(info)
        if self.phase == 2:
            # 贴住砖堆后，重踢打得比拳头更远，能拆掉拳头够不着的一层。
            return self._mash(info, BUTTON_C, period=22, quiet=90, crouch=False, cap=1100)
        # 剩下的中心用中拳拆完。
        return self._mash(info, BUTTON_Y, period=14, quiet=70, crouch=False, cap=900)

    def _step_into_bricks(self, info):
        """收招结束后往右走到被砖挡住为止。"""
        t = self.phase_frame
        self.phase_frame += 1
        x = int(info.get("agent_x", 0))
        action = _blank_action()
        if t < 10:
            self.prev_x = x
            self.still = 0
            return action
        action[BUTTON_RIGHT] = 1
        if self.prev_x is not None and x == self.prev_x:
            self.still += 1
        else:
            self.still = 0
        self.prev_x = x
        if self.still > 10 or t > 60:
            self._begin_phase(2)
        return action

    def _barrel(self, frame, info, screen):
        """Ken 打木桶的打法，用在龙身上。

        轻拳升龙。桶还在滑槽高处时不出手，等它落到拳头高度，节拍大约 29 帧一发。
        这一发会把人往前带。带到滑槽右边以后先走回来，这段就是中途故意等的那一下，
        免得人追出落点、后面的桶接不住。
        已经滚到左边、而滑槽里又没有正掉下来的桶时，下一发改朝左。
        没有画面时退回只朝右的节拍，至少能打碎滑槽正下方的那些。
        """
        if self.barrel_age is not None:
            action = self._barrel_motion()
            self.barrel_prev = screen
            return action
        action = _blank_action()
        if screen is None or frame < 280:
            self.barrel_prev = screen
            if screen is None and frame >= 280:
                return self._barrel_beat(frame, BUTTON_RIGHT)
            return action
        ay = int(info.get("agent_y", 0))
        ax = int(info.get("agent_x", 0))
        if ay < 186 or frame < self.barrel_ready or self.barrel_prev is None:
            self.barrel_prev = screen
            return action
        rx = 73 + (ax - 205) * (10 / 15.0)
        right_falling, left_ground = _barrels_near(self.barrel_prev, screen, rx)
        self.barrel_prev = screen
        # 升龙把人带过滑槽之后，先走回落点下面再发下一发。
        if ax > 255:
            action[BUTTON_LEFT] = 1
            return action
        if (frame - 280) % 29 != 0:
            return action
        forward = BUTTON_LEFT if left_ground and not right_falling else BUTTON_RIGHT
        self.barrel_forward = forward
        self.barrel_age = 0
        self.barrel_ready = frame + 26
        return self._barrel_motion()

    def _barrel_motion(self):
        """一发轻拳升龙：前、下、下前、下前加轻拳，每步一帧。"""
        age = self.barrel_age
        self.barrel_age += 1
        if self.barrel_age > 3:
            self.barrel_age = None
        action = _blank_action()
        forward = self.barrel_forward
        if age == 0:
            action[forward] = 1
        elif age == 1:
            action[BUTTON_DOWN] = 1
        elif age == 2:
            action[BUTTON_DOWN] = 1
            action[forward] = 1
        elif age == 3:
            action[BUTTON_DOWN] = 1
            action[forward] = 1
            action[BUTTON_X] = 1
        return action

    def _barrel_beat(self, frame, forward):
        """没有画面时，按固定节拍朝一边出轻拳升龙。"""
        action = _blank_action()
        t = (frame - 280) % 29
        if t == 0:
            action[forward] = 1
        elif t == 1:
            action[BUTTON_DOWN] = 1
        elif t == 2:
            action[BUTTON_DOWN] = 1
            action[forward] = 1
        elif t == 3:
            action[BUTTON_DOWN] = 1
            action[forward] = 1
            action[BUTTON_X] = 1
        return action


def result_skip_action(frames_since_tick, skip_done):
    """分数开始跳之后满 1 秒，点一次轻拳。没跳过去就再补一次。"""
    action = np.zeros(12, dtype=np.int8)
    if skip_done or frames_since_tick is None:
        return action
    # 第一次在 1 秒时。有的结算第一下没吃到，半秒后再点一下。
    for start in (RESULT_HOLD_FRAMES, RESULT_HOLD_FRAMES + 30):
        if start <= frames_since_tick < start + RESULT_TAP_FRAMES:
            action[BUTTON_X] = 1
            break
    return action


class SFighterCustomWrapper(gym.Wrapper):
    """把一局街机对战变成强化学习能用的观测、奖励和结束信号。"""

    def __init__(self, env, obs_size=84, render=False, initial_level=0,
                 time_penalty=0.0005,
                 render_skip=1, reset_round=False, max_episode_steps=18000,
                 realtime=False):
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

        胜利台词和世界地图保持空按键。进入 bonus 后改用固定套路出拳，
        不把这种关卡交给只见过对战画面的模型。
        双方血量回到 100 以上，而且场景仍是对战场地，才交还模型。
        """
        info = {}
        announced_bonus = set()
        for _ in range(max_steps):
            if self.render_enabled:
                self.render_game()
            stage = info.get("stage")
            stage_id = int(stage) if stage is not None else -1
            in_bonus = stage_id > 11
            if in_bonus:
                if stage_id not in announced_bonus:
                    announced_bonus.add(stage_id)
                    name = BONUS_STAGE_NAMES.get(stage_id, str(stage_id))
                    print(f"Bonus stage: {name}. Using a fixed routine.", flush=True)
                # 木桶要看桶落在哪。这里拿的是模拟器原始画面，不是模型用的那张缩小图。
                screen = self.env.env.render()
                action = self._bonus.action(stage_id, info, screen)
            else:
                action = NO_OP_ACTION
            obs, _, _, truncated, info = self.env.step(action)
            if truncated:
                raise RuntimeError(
                    "Game was truncated while waiting for the next match"
                )
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
            )
        frame = self.env.render()
        # get_audio() 取走的是刚刚这一帧模拟器产生的立体声。
        audio = self.env.unwrapped.em.get_audio()
        self._presenter.present(frame, audio)

class BarrelBonusWrapper(SFighterCustomWrapper):
    """木桶专用的训练环境。画面处理跟对战一样，奖励改成实际加上的分数。

    出招在外层锁成轻拳升龙。分数会奖励打中的时机、打碎的只数，以及收得更快的时间奖励。
    """

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        self.waiting_for_next_round = False
        self._match_decided = False
        self._motion_frames = collections.deque(maxlen=5)
        # 木桶存档是模拟器内存快照，不是对战关那种 gzip。读进来之后直接灌进模拟器。
        if not hasattr(self, "_barrel_state"):
            with open(BARREL_STATE_PATH, "rb") as save:
                self._barrel_state = save.read()
        self.env.reset(**kwargs)
        emulator = self.env.unwrapped.em
        emulator.set_state(self._barrel_state)
        obs = self.env.render()
        info = self.env.unwrapped.data.lookup_all()
        obs = self._process_frame(obs)
        self.frame_stack.clear()
        self.frame_stack.extend([obs] * self.stack_size)
        self._score_at_reset = int(info.get("score", 0))
        self._prev_score = self._score_at_reset
        return self._stack_observation(), info

    def step(self, action):
        """一个模型动作仍是 4 帧按键。分数在这 4 帧里加上多少，奖励就是多少。"""
        self.elapsed_steps += 1
        info = {}
        for frame_action in action:
            obs, _, _, _, info = self.env.step(frame_action)
        obs = self._process_frame(obs)
        self.frame_stack.append(obs)
        if "score" not in info or "stage" not in info:
            info = self.env.unwrapped.data.lookup_all()
        score = int(info.get("score", self._prev_score))
        # 结算偶尔会把分数一次加上，不会往下掉。往下的跳动不当成惩罚。
        gained_now = max(0, score - self._prev_score)
        self._prev_score = score
        # 一千分收成 1，和一次对战里打掉一截血的奖励差不多大。
        # 时间惩罚沿用对战的 0.0005，避免干站着等桶自己滚走。
        reward = gained_now * 0.001 - self.time_penalty
        stage = int(info.get("stage", BONUS_BARREL))
        # 木桶结束、游戏切去下一场时，这一局的分已经加完。
        done = stage != BONUS_BARREL
        truncated = (not done) and self.elapsed_steps >= self.max_episode_steps
        info["barrel_gained"] = score - self._score_at_reset
        return self._stack_observation(), float(reward), done, truncated, info


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


class BarrelActionWrapper(gym.ActionWrapper):
    """木桶只允许五件事：站着等、往左走、往右走、朝左轻拳升龙、朝右轻拳升龙。

    打碎桶的招式锁成轻拳升龙。模型要学的是站到桶前面，以及等它落到拳头高度再发。
    一次决策仍是 4 帧，和升龙的指令长度一致，也和对战观测的采样间隔一致。
    """

    def __init__(self, env):
        super().__init__(env)
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
                   'C', 'Y', 'X', 'Z')

        def frame(*pressed):
            return [int(button in pressed) for button in buttons]

        def shoryuken(forward):
            # 前、下、下前、下前加轻拳。轻拳是 X，不是对战升龙里用的中拳。
            return [
                frame(forward),
                frame('DOWN'),
                frame('DOWN', forward),
                frame('DOWN', forward, 'X'),
            ]

        self._actions = [
            [frame()] * 4,
            [frame('LEFT')] * 4,
            [frame('RIGHT')] * 4,
            shoryuken('LEFT'),
            shoryuken('RIGHT'),
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]
