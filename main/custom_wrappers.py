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
import random

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


def barrel_state_paths(include_original=True):
    """训练用的木桶开局。原来的存档可以不放进来，只练实战截下来的三份。

    木桶落点每次不一样。只在一份存档上练，换一局里就会打砸。
    每局开始随机抽一份，让同一套出招见到不同的桶。
    """
    paths = [BARREL_STATE_PATH] if include_original else []
    prefix = "Champion.X.BonusBarrel.live"
    for name in sorted(os.listdir(DATA_DIR)):
        if name.startswith(prefix) and name.endswith(".state"):
            paths.append(os.path.join(DATA_DIR, name))
    if not paths:
        paths.append(BARREL_STATE_PATH)
    return paths
def _blank_action():
    return np.zeros(12, dtype=np.int8)


def _barrel_blobs(prev, img, ryu_x):
    """和上一帧对不上的色块，相对人的位置。用来判断这一发升龙朝没朝向桶。"""
    if prev is None or img is None or prev.shape != img.shape:
        return []
    diff = np.abs(img.astype(np.int16) - prev.astype(np.int16)).sum(axis=2)
    mask = (diff > 110).astype(np.uint8)
    mask[:40] = 0
    mask[190:] = 0
    mask[mask.sum(axis=1) > 50] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        if not (40 <= area <= 4000 and height >= 8 and width <= 90):
            continue
        x, y = centroids[i]
        found.append((float(x - ryu_x), float(y), int(area)))
    return found


def _barrel_points(prev, img, ryu_x):
    """正在往下掉、以及落地后横着滚的桶，在画面上的位置。

    人自己走动的身体不算桶。贴地的桶也留下，落地之后还要打碎。
    整行一起闪的是背景动画，不算。
    """
    if prev is None or img is None or prev.shape != img.shape:
        return []
    diff = np.abs(img.astype(np.int16) - prev.astype(np.int16)).sum(axis=2)
    mask = (diff > 110).astype(np.uint8)
    mask[:40] = 0
    mask[210:] = 0
    mask[mask.sum(axis=1) > 50] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        if not (40 <= area <= 4000 and height >= 8 and width <= 90):
            continue
        x, y = float(centroids[i][0]), float(centroids[i][1])
        # 人站在地上时，身体也会和上一帧对不上。离人这么近的一块不当成桶。
        if abs(x - ryu_x) < 22 and y > 160:
            continue
        found.append((x, y))
    return found


class BonusRoutine:
    """三关 bonus 的固定套路。不看模型，只看分数有没有还在涨、人站在哪。

    砸汽车：左边贴身一记重腿。龙和肯贴得很近时这一脚自己打两段，
    接着重拳升龙翻过车顶落到右边，上半截用重踢，底盘蹲着打。
    打砖墙：一能打中就重拳升龙，先上到左侧高处，把还能打中的重腿踢完。
    再走到右边地面踢。半路和多出来的那一脚都没伤害，不出。
    滚木桶不走这里，连续闯关用和三份存档同一套规则。
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

    def action(self, stage, info, screen=None):
        """返回这一帧的 12 键。人还没站到场上时什么都不按。

        screen 是汽车和砖墙用的原始画面。木桶不经过这里。
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
        return _blank_action()

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
        # 前三百帧人还没站稳，重腿和升龙都打不中。
        if frame < 300:
            return _blank_action()
        if self.phase == 0:
            # 一能打中就升龙。先打掉靠上的砖，人落到墙的高处，不等下面踢完。
            return self._brick_mount(info)
        if self.phase == 1:
            # 人在左侧高处。这里连踢三脚都有二段，第四脚是空的，踢完再往右走。
            return self._brick_kicks(3, 2)
        if self.phase == 2:
            # 左边能打中的都打完了。走到右边地面再踢，走的途中不出手。
            return self._brick_descend(info)
        if self.phase == 4:
            # 先转向面对砖墙，再踢出二段。
            return self._brick_right_kicks()
        # 还没拆完就再朝砖堆升龙贴近，然后继续重腿。
        return self._brick_close_in(info)

    def _brick_kick_frame(self, t):
        """一记贴身重腿。按下两帧，剩下的帧把两段伤害和收招让完。"""
        action = _blank_action()
        if t < 2:
            action[BUTTON_C] = 1
        return action

    def _brick_kicks(self, count, next_phase):
        """连踢 count 脚。一脚 70 帧，短于这个间隔，下一脚会踢在收招里。"""
        t = self.phase_frame
        self.phase_frame += 1
        if t >= count * 70:
            self._begin_phase(next_phase)
            return _blank_action()
        return self._brick_kick_frame(t % 70)

    def _brick_dp_frame(self, t, forward):
        """重拳升龙。每个方向按满两帧才会离地。"""
        action = _blank_action()
        part = t // 2
        if part == 0:
            action[forward] = 1
        elif part == 1:
            action[BUTTON_DOWN] = 1
        else:
            action[BUTTON_DOWN] = 1
            action[forward] = 1
            if part == 3:
                action[BUTTON_Z] = 1
        return action

    def _brick_mount(self, info):
        """开打后的第一记重拳升龙，把人送到砖堆高处，不等踢完外层。"""
        t = self.phase_frame
        y = int(info.get("agent_y", 192))
        status = int(info.get("agent_status", 0))
        if t < 8:
            self.phase_frame += 1
            return self._brick_dp_frame(t, BUTTON_RIGHT)
        self.phase_frame += 1
        if y < 140:
            self.crossed = True
        # 落到墙的高处就停。地面是 192，这里 y 明显更小才算在墙上。
        if self.crossed and t > 40 and status == 512 and y <= 175:
            self._begin_phase(1)
            return _blank_action()
        action = _blank_action()
        action[BUTTON_RIGHT] = 1
        return action

    def _brick_chase(self, info):
        """朝右升龙，落到砖堆里面，收招完再把控制权交给下一串重腿。"""
        t = self.phase_frame
        self.phase_frame += 1
        if t < 8:
            return self._brick_dp_frame(t, BUTTON_RIGHT)
        y = int(info.get("agent_y", 192))
        if y < 180:
            self.crossed = True
        if self.crossed and y >= 190 and t > 40:
            self.still += 1
        else:
            self.still = 0
        # 落地后状态还停在升龙上，空几帧再踢，否则这一脚不出伤害。
        if self.still >= 10:
            self._begin_phase(2)
        return _blank_action()

    def _brick_descend(self, info):
        """走到砖堆右边的地面再转身。半路踢是空的，所以这段只走路。"""
        y = int(info.get("agent_y", 192))
        x = int(info.get("agent_x", 0))
        if self.still == 0:
            self.phase_frame += 1
            # 走到 x=335 左右、双脚在地面时，朝左的重腿才又有二段。
            if (y >= 190 and x >= 310) or self.phase_frame > 180:
                self.still = 1
                self.phase_frame = 0
            else:
                action = _blank_action()
                action[BUTTON_RIGHT] = 1
                return action
        self.phase_frame += 1
        if self.phase_frame >= 10:
            self._begin_phase(4)
            return _blank_action()
        action = _blank_action()
        action[BUTTON_LEFT] = 1
        return action

    def _brick_cross(self, info):
        """人已在高处。再一记重拳升龙打掉最顶上的砖，落到最高处后走到右边下去。"""
        y = int(info.get("agent_y", 192))
        x = int(info.get("agent_x", 0))
        if self.still == 0 and self.phase_frame == 0:
            # 追击那一跳留下的离地标记不能算这次已经落地。
            self.crossed = False
        if self.still == 0:
            t = self.phase_frame
            if t < 8:
                self.phase_frame += 1
                return self._brick_dp_frame(t, BUTTON_RIGHT)
            self.phase_frame += 1
            if y < 160:
                self.crossed = True
            # 墙顶大约 123。走到右端会自己落到地面，不要在墙顶上踢。
            if not (self.crossed and y >= 190 and x >= 300) and self.phase_frame < 240:
                action = _blank_action()
                action[BUTTON_RIGHT] = 1
                return action
            self.still = 1
            self.phase_frame = 0
        self.phase_frame += 1
        if self.phase_frame >= 10:
            self._begin_phase(4)
            return _blank_action()
        action = _blank_action()
        action[BUTTON_LEFT] = 1
        return action

    def _brick_right_kicks(self):
        """落到右边地面后先转向，再踢三脚二段。第四脚没伤害，不踢。"""
        t = self.phase_frame
        self.phase_frame += 1
        if t < 10:
            action = _blank_action()
            action[BUTTON_LEFT] = 1
            return action
        kick_t = t - 10
        if kick_t >= 3 * 70:
            # 第四脚是空的。剩下的砖再贴过去打。
            self._begin_phase(5)
            self.still = 1
            return _blank_action()
        return self._brick_kick_frame(kick_t % 70)

    def _brick_close_in(self, info):
        """右边三脚之后接着踢。踢空了就朝砖跳过去，不升龙。升龙这段打不中。"""
        y = int(info.get("agent_y", 192))
        x = int(info.get("agent_x", 0))
        if not self.saw_score:
            # 刚换到这一段时人已经贴着砖，先踢。踢空之后 still 被置上，才去移位。
            if self.still == 0:
                self.saw_score = True
                self.prev_score = int(info.get("score", 0))
                self.phase_frame = 0
            else:
                return self._brick_step_closer(info, x, y)
        t = self.phase_frame
        self.phase_frame += 1
        if t >= 70:
            score = int(info.get("score", 0))
            if score == self.prev_score:
                self.saw_score = False
                self.phase_frame = 0
                self.crossed = False
                self.still = 1
                self.quiet += 1
            else:
                self.prev_score = score
                self.phase_frame = 0
                self.quiet = 0
            return _blank_action()
        return self._brick_kick_frame(t)

    def _brick_step_closer(self, info, x, y):
        """离砖还远就朝砖跳上矮台。重旋风腿和升龙过去都不上分，跳更短。"""
        t = self.phase_frame
        far = x > 318
        status = int(info.get("agent_status", 0))
        if far:
            if t < 2:
                self.phase_frame += 1
                action = _blank_action()
                # 朝砖的方向跳。落到 y=164 左右的矮台上，重腿又能打出二段。
                action[BUTTON_LEFT] = 1
                action[BUTTON_UP] = 1
                return action
            self.phase_frame += 1
            if y < 170:
                self.crossed = True
            if (self.crossed and t > 20 and status == 512 and y <= 180) or t > 80:
                self.saw_score = True
                self.phase_frame = 0
                self.prev_score = int(info.get("score", 0))
                self.crossed = False
            return _blank_action()
        # 近处空踢说明差一点点。按住朝砖的方向几帧，比再升一次龙省时间。
        if t < 8:
            self.phase_frame += 1
            action = _blank_action()
            action[BUTTON_LEFT] = 1
            return action
        self.saw_score = True
        self.phase_frame = 0
        self.prev_score = int(info.get("score", 0))
        return _blank_action()


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

# 站立和走路都是 512。522 是轻拳。524 是升龙，也是波动拳，两者靠有没有离地分开。
# 朝当前脸的方向，升龙第 16 帧离地。面朝左再朝右打，要到第 30 帧才离地。
# 波动拳一直站在地上，大约第 55 帧才回到站立。第 40 帧还没离地，才当成波动拳。
BARREL_IDLE_STATUS = 512
BARREL_LIGHT_PUNCH_STATUS = 522
BARREL_SPECIAL_STATUS = 524
BARREL_GROUND_Y = 190
BARREL_GROUND_SPECIAL_FRAMES = 40
# 打空一次扣 1，记在发出这下升龙的同一步。扣到 4 和 8 时，扣分落在落地后的另一步，
# 模型只少挥几下，过关奖励丢了，并没有改成走位。打中一千分仍是 10，比打空划算。
BARREL_WHIFF_PENALTY = 1.0
BARREL_WRONG_MOVE_PENALTY = 2.0
# 每一千分给 10。随机出手一局大约一万七千分，同时打空、发错和罚站大约扣 76。
# 按每一千分给 1 算，加分只有 17，干站着只扣大约 38。给 10 之后出手才盖过扣分。
BARREL_SCORE_SCALE = 0.01
# 朝反方向打中不再加分。加分会让模型先朝没有桶的一侧挥一下，只为了换边。
BARREL_TURN_BONUS = 0.0
# 收招没完就再按，这一下被吃掉。只轻轻扣一点，避免完全不出手比乱按更赚。
BARREL_BLOCKED_PENALTY = 0.05
# 游戏大约 60 帧为 1 秒。站在落点下等桶是该做的事，不再因为这一秒没加分而扣。
BARREL_SECOND_FRAMES = 60
BARREL_IDLE_SECOND_PENALTY = 0.0
# 画面差出来的色块对不准「这一下会不会打中」。
# 用它来扣分会把已经打穿的存档一起罚掉，所以先记着，不扣。
BARREL_UNAIMED_PENALTY = 0.0
# 被桶打倒是 526。扣 8 分时，另外两份已经清完的存档在两万五千步里就丢了过关奖励，
# 这份没清完的也没有因此清完，所以先不扣。状态仍记着，确认新的扣分值再打开。
BARREL_KNOCKDOWN_STATUS = 526
BARREL_KNOCKDOWN_PENALTY = 0.0
# 还在空中的桶，瞄准它掉到拳头高度的那一列，争取落地前打碎。
# 已经贴地横滚的桶也要追，瞄准它现在的位置。人还没走到拳头范围，势是剩下的距离。
# 每步用和 PPO 相同的 γ=0.99：γ×新势−旧势，只对同一个桶。
# 走近几像素大约零点几分，低于打碎一个桶的 10，避免只走路不打。
BARREL_APPROACH_SCALE = 0.06
BARREL_APPROACH_MARGIN = 24.0
BARREL_SHAPING_GAMMA = 0.99
BARREL_SHAPING_CLIP = 0.5
BARREL_FALL_MIN_SPEED = 0.35
BARREL_TRACK_MIN_SAMPLES = 4
BARREL_TRACK_MIN_FALLING = 3
BARREL_TRACK_MATCH = 36.0
BARREL_LAND_Y = 185.0
BARREL_SAMPLE_FRAMES = 4
# 升龙指令只有 6 帧，人要更晚才落地。这一步一直模拟到落地，打中和打空才记在出招的这一步上。
BARREL_MOVE_HOLD_FRAMES = 96
BARREL_SCORE_SETTLE_FRAMES = 12
# 波动拳收回来之后，火球还要再飞一阵才打到远处正在滚的桶。这段也算进发出这一下的同一步。
BARREL_HADOU_SETTLE_FRAMES = 72
# 拳头够得到的高度。桶还在这之上时，瞄准它掉到这条线的那一列。
BARREL_STRIKE_Y = 130.0
# 贴地之后圆桶的外框中心停在 168 到 169。167 是这条线的下沿。
# 还在往下掉的外框到不了 166，不能当成已经在滚。
BARREL_FLOOR_Y = 167.0
# 模型看到最多三只桶的位置和速度，再加上人的横坐标和朝向。不是画面。
BARREL_STATE_SLOTS = 3
BARREL_STATE_DIM = BARREL_STATE_SLOTS * 8 + 2
# 轻升龙够得到的距离。更远就先走，不用升龙去挪位置。
BARREL_FIST_RANGE = 36.0
# 站定后往一侧走，大约每帧 0.7 个画面像素。
BARREL_WALK_SPEED = 0.7
# 开局先走到落桶区域。出生点在画面左侧。112 是落点中位，就停在中心。
BARREL_HOME_X = 112.0
# 轻升龙拳头大约到这个高度。桶还在这之上就先走位，进了这个高度再出招。
BARREL_PUNCH_Y = 165.0
# 滑槽里的桶每帧大约只掉 1 到 2 像素，横坐标还会顺着槽来回晃。
# 竖直速度超过这个，才算已经离开滑槽，左右才定得下来。
BARREL_DROP_VY = 2.5
# 第一招加分的前一帧，桶还在，桶心在 48，拳头刚好打到。这是升龙打得到的最高处。
# 155 是桶已经落到胸口附近，拿它当最高点会一直等到桶掉下去。
BARREL_APEX_Y = 48.0
# 按下到拳头升到最高点大约 28 帧。桶心停在 38 到 46、还没有明显往下掉，现在就要按。
# 已经每帧掉超过 0.8 像素的，28 帧后桶在胸口，拳头打在最高点的空处。
# 往上弹的那一下速度是负的，不受这道限制，照样按。
BARREL_APEX_PRESS_LO = 38.0
BARREL_APEX_HANG_VY = 0.8
# 有的桶在 33 就到了头旁边。这一档也按，不等它落到 38。
# 还在槽中间、离头超过一个桶半径的，不在这里按。
BARREL_APEX_NEAR_LO = 32.0
BARREL_APEX_NEAR_X = 18.0
BARREL_APEX_PRESS_HI = 46.0
# 轻升龙从按下到拳头进入这条高度大约 15 帧，之后一小段也还打得着。
# 桶还要 8 到 18 帧才到这条线，现在出招，是最早能打上的时机。再早，升龙会先落地。
BARREL_DP_LEAD_MIN = 8.0
BARREL_DP_LEAD_MAX = 18.0
# 离开滑槽后大约每帧掉 4.5 像素。拳头高度再倒推 15 帧，就是该按下的高度。
# 实测第一只桶从 y≈86 掉到 y≈158 用了 16 帧，和升龙起手对齐。
BARREL_DROP_SPEED = 4.5
# 一发轻升龙身体大约挪 9 个画面像素。只和地上滚动的桶比这个水平距离，不看高度。
# 滚得比这一发更远，才用波动拳追。
BARREL_DP_SHIFT = 10.0
# 最下面的桶外框已经低于这条线、横向又超过 45 时，先看后面两帧往哪边滚再出招。
# 朝人滚出升龙，往远处滚出轻拳波动。没超过 45 不看方向，直接升龙。
# 外框中心贴地后在 168 到 169。169 是按旧的热度中心定的，那一点会跟着桶板纹路掉到 167 以下。
BARREL_ROLL_Y = 167.0
BARREL_FALL_DP_X = 45.0
BARREL_HADOU_Y = 140.0
# live1 走到中间后，第 87 帧朝左升龙打不中，桶的色块在 y=69。
# 再过一帧才打出 300，而且要朝桶滑过来的方向。以后第一发就出在这个高度的下一帧。
BARREL_OPEN_Y = 69.0
# 升龙脚在第 43 帧落地，状态仍是 524，第 51 帧才回到站立。
# 同向下一发第 46 帧就能按，反向要到第 50 帧。先在 46 交还，方向相反再补到 50。
BARREL_DP_CANCEL_SAME = 46
BARREL_DP_CANCEL_FRAME = 50
# 波动拳第 53 帧回到站立。同向升龙第 49 帧能接上，反向升龙第 53 帧能接上。
# 紧接着再发波动拳，中拳落在第 79 帧才出得来，更早只会挥成轻拳。
BARREL_HADOU_DP_CANCEL_FRAME = 53
BARREL_HADOU_CHAIN_PUNCH = 79


class BarrelBonusWrapper(SFighterCustomWrapper):
    """木桶专用的训练环境。奖励是游戏实际加上的分数。

    旧模型看运动图，只出轻拳升龙。新模型看桶的位置：还在空中的，以及落地后横着滚的。
    出招多了左右波动拳。升龙和波动拳都要等这下收完，打空记在发出的这一步上。
    走近该处理的那只桶有一小笔势。站着等桶掉下来不扣分。
    """

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        self.waiting_for_next_round = False
        self._match_decided = False
        self._motion_frames = collections.deque(maxlen=5)
        # 木桶存档是模拟器内存快照，不是对战关那种 gzip。读进来之后直接灌进模拟器。
        # 实战截下来的开局和原始存档放在一起，每局随机抽一份。
        if not hasattr(self, "_barrel_library"):
            self._barrel_paths = barrel_state_paths(
                include_original=not getattr(self, "live_only", False)
            )
            self._barrel_library = []
            for path in self._barrel_paths:
                with open(path, "rb") as save:
                    self._barrel_library.append(save.read())
        forced = getattr(self, "force_barrel_index", None)
        if forced is None:
            index = random.randrange(len(self._barrel_library))
        else:
            index = int(forced) % len(self._barrel_library)
        self._barrel_state = self._barrel_library[index]
        self._barrel_state_name = os.path.basename(self._barrel_paths[index])
        self.env.reset(**kwargs)
        emulator = self.env.unwrapped.em
        emulator.set_state(self._barrel_state)
        obs = self.env.render()
        info = self.env.unwrapped.data.lookup_all()
        obs = self._process_frame(obs)
        self.frame_stack.clear()
        self.frame_stack.extend([obs] * self.stack_size)
        self._arm_rule(info)
        return self._decision_observation(), info

    def begin_live(self):
        """连续闯关已经走到木桶，人站在场上。不读存档，用和三份存档同一套规则打。"""
        self.elapsed_steps = 0
        self.waiting_for_next_round = False
        self._match_decided = False
        self._motion_frames = collections.deque(maxlen=5)
        self.frame_stack.clear()
        screen = self.env.render()
        if screen is not None:
            obs = self._process_frame(np.array(screen, copy=True))
            self.frame_stack.extend([obs] * self.stack_size)
        info = self.env.unwrapped.data.lookup_all()
        self._barrel_state_name = "arcade"
        self.max_episode_steps = max(int(self.max_episode_steps), 20000)
        self._arm_rule(info)
        # 飞进来之后人已经不在出生点。先看一眼再走，免得第一拍按出生点的坐标往回走。
        self._since_sample = BARREL_SAMPLE_FRAMES
        self._sample_tracks(self.env.unwrapped.data.lookup_all())

    def _arm_rule(self, info):
        """出招状态清零。读存档和从连续闯关走进来都从这里开始。"""
        self._score_at_reset = int(info.get("score", 0))
        self._prev_score = self._score_at_reset
        # 正在判定的这一下升龙。人回到站立、或确认不是升龙时才结算。
        self._move = None
        # 上一发真正送出去的升龙朝哪边。下一发方向相反而且打中时，这一下的分数再给一倍。
        self._last_dp_forward = None
        # 换边的升龙落地后，分数可能晚几帧才跳。这段帧里的加分也算这一下。
        self._boost_frames = 0
        # 第 1 秒结束时的分数。用来判断接下来每一秒有没有打出伤害。
        self._frames = 0
        self._second_score = self._score_at_reset
        # 上一帧原始画面。和当前帧对一下，才知道桶在人的哪一侧。
        self._aim_prev = None
        self._down = False
        self._tracks = []
        self._vision_tracks = False
        self._next_track = 1
        self._track_prev = None
        self._since_sample = 0
        self._approach = None
        # 实战开局面朝右。走路和出招会改这个，位置向量里要把脸告诉模型。
        self._facing = 1
        self._rx = 73.0
        self._body_x = 73.0
        self._last_intend = None
        self._pending_whiff = 0.0
        self._rule_last = 0
        self._rule_attack_score = None
        # 还没走到落点中间。走到一次之后，被撞开也不再退回只走路。
        self._rule_home = False
        self._rule_seen_line = False
        self._rule_opened = False
        self._rule_open_side = None
        self._rule_cancel = False
        self._handed_frames = 0
        self._handed_intend = None
        self._roll_pending = None
        from main.barrel_vision import forget_frame
        forget_frame()

    @staticmethod
    def new_move_gate(score):
        """只带着出招门闩，不启动模拟器。演示打木桶时和训练用同一套结算。"""
        gate = BarrelBonusWrapper.__new__(BarrelBonusWrapper)
        gate._move = None
        gate._last_dp_forward = None
        gate._boost_frames = 0
        gate._boost_from = 0
        gate._prev_score = int(score)
        return gate

    def _attack_pressed(self, action):
        """这几帧里有没有按下轻拳或中拳。只有升龙的最后一帧会按。"""
        return any(int(frame[BUTTON_X]) or int(frame[BUTTON_Y]) for frame in action)

    def _shoryuken_ready(self, info):
        """只有站在地上、没有别的招没收完，升龙指令才会被游戏收成升龙。"""
        status = int(info.get("agent_status", 0))
        height = int(info.get("agent_y", 0))
        return status == BARREL_IDLE_STATUS and height >= BARREL_GROUND_Y and self._move is None

    def _can_cancel_into(self, info):
        """上一发还没回到站立，但已经到了试验过能接下一招的帧。"""
        move = self._move
        if not getattr(self, "_rule_cancel", False) or move is None:
            return False
        if int(info.get("agent_y", 0)) < BARREL_GROUND_Y:
            return False
        need = BARREL_DP_CANCEL_FRAME if move.get("intend") != "hadou" else BARREL_HADOU_DP_CANCEL_FRAME
        return int(move.get("frames", 0)) >= need

    def _dp_forward(self, action):
        """出招那一帧按住的方向就是朝向。波动拳第一帧是下，方向在按拳的那一帧。"""
        for frame in action:
            if int(frame[BUTTON_X]) or int(frame[BUTTON_Y]):
                if int(frame[BUTTON_LEFT]):
                    return BUTTON_LEFT
                if int(frame[BUTTON_RIGHT]):
                    return BUTTON_RIGHT
        return BUTTON_LEFT if int(action[0][BUTTON_LEFT]) else BUTTON_RIGHT

    def _shoryuken_aimed(self, info, forward, screen):
        """这一侧看得到正在动的桶，才算出手对准了。

        上一手开始时的画面和现在这一帧对一下。两帧没差别，说明桶还没动到能打的位置。
        出生点上桶在身前，朝左打会挥空。朝右要身前有色块，朝左要左侧有色块。
        还没有上一帧，或者画面上没有色块，不罚。
        """
        prev = self._aim_prev
        ax = int(info.get("agent_x", 0))
        rx = 73 + (ax - 205) * (10 / 15.0)
        found = _barrel_blobs(prev, screen, rx)
        if not found:
            return True
        # 看离人最近的那一块。身前的桶朝左打会空，左侧近处的桶朝右打也会空。
        dx, y, _area = min(found, key=lambda blob: abs(blob[0]))
        if forward == BUTTON_RIGHT:
            return -15 <= dx <= 45 and 45 <= y <= 175
        return -60 <= dx <= -8 and y >= 50

    def _begin_move(self, info, forward, intend="dp"):
        # 和上一发朝向不同，这一下打中的分才再给一倍。第一发没有上一发，不加成。
        # intend 是这次想出的招。升龙没离地、收成波动拳，才算发错。主动出的波动拳不是发错。
        switched = self._last_dp_forward is not None and forward != self._last_dp_forward
        self._last_dp_forward = forward
        self._last_intend = intend
        self._facing = -1 if forward == BUTTON_LEFT else 1
        self._move = {
            "score": int(info.get("score", self._prev_score)),
            "frames": 0,
            "air": False,
            "light": False,
            "switched": switched,
            "intend": intend,
            "saw_ground": False,
        }

    def _observe_move(self, info):
        """看着这一下到底升空没有、有没有变成轻拳。招式结束时返回要扣的分。"""
        if self._move is None:
            return 0.0
        status = int(info.get("agent_status", 0))
        height = int(info.get("agent_y", BARREL_GROUND_Y))
        score = int(info.get("score", self._move["score"]))
        self._move["frames"] += 1
        if height < BARREL_GROUND_Y:
            self._move["air"] = True
        # 轻拳和中拳打空都是 522。升龙没做成、只挥出一拳时走这条。
        if status == BARREL_LIGHT_PUNCH_STATUS:
            self._move["light"] = True
        kind = None
        if self._move["light"]:
            kind = "light"
        elif self._move["air"] and status == BARREL_IDLE_STATUS and height >= BARREL_GROUND_Y:
            kind = "dp"
        elif (
            not self._move["air"]
            and status == BARREL_SPECIAL_STATUS
            and self._move["frames"] >= BARREL_GROUND_SPECIAL_FRAMES
        ):
            # 波动拳一直站在地上。先记住，等收招回到站立再结算，避免火球还没飞出去就判打空。
            self._move["saw_ground"] = True
        if self._move.get("saw_ground") and status == BARREL_IDLE_STATUS and height >= BARREL_GROUND_Y:
            kind = "hadou"
        elif self._move.get("saw_ground") and self._move["frames"] > 80:
            kind = "hadou"
        elif kind is None and self._move["frames"] > 80:
            # 指令送出去以后人始终没离开站立，游戏没收下这一招。
            kind = "ignored"
        if kind is None:
            return 0.0
        gained = score - self._move["score"]
        intend = self._move.get("intend", "dp")
        switched = bool(self._move["switched"])
        self._move = None
        came_out = (intend == "dp" and kind == "dp") or (intend == "hadou" and kind == "hadou")
        # 想出的招真的出来了，这段里分数涨了，就不扣。升龙换边打中仍按原来的窗口补分。
        if came_out and gained > 0:
            if intend == "dp" and switched:
                self._boost_frames = 24
                self._boost_from = score
                return -gained * BARREL_SCORE_SCALE * BARREL_TURN_BONUS
            return 0.0
        # 波动拳打中往往在收招之后。先挂着，这一步结束时分数真的没涨再扣。
        if came_out and intend == "hadou":
            self._pending_whiff = BARREL_WHIFF_PENALTY
            return 0.0
        if came_out or kind == "ignored":
            return BARREL_WHIFF_PENALTY
        if gained > 0:
            return 0.0
        return BARREL_WRONG_MOVE_PENALTY

    def _idle_second_penalty(self, info):
        """从第 2 秒起，刚过去的这一秒里分数没有上涨，就扣一次。"""
        self._frames += 1
        if self._frames % BARREL_SECOND_FRAMES != 0:
            return 0.0
        score = int(info.get("score", self._second_score))
        # 第 1 秒只记住分数，不罚。桶还在高处，人打不中。
        if self._frames < 2 * BARREL_SECOND_FRAMES:
            self._second_score = score
            return 0.0
        idle = score <= self._second_score
        self._second_score = score
        if idle:
            return BARREL_IDLE_SECOND_PENALTY
        return 0.0

    def _sample_tracks(self, info):
        """隔几帧看一次桶。往下掉的和贴地横滚的都留下。

        某一拍网络没再报出这个点，正在掉或正在滚的多留两拍，其余的下一拍就丢掉。
        这只是漏看时的兜底。墙上的桶被认成目标，要改识别，不在这里滤。
        """
        screen = self.env.render()
        prev = self._track_prev
        self._track_prev = None if screen is None else np.array(screen, copy=True)
        ax = int(info.get("agent_x", 0))
        # 走到中间仍用这条压缩过的横坐标，落点不用重调。
        rx = 73 + (ax - 205) * (10 / 15.0)
        self._rx = rx
        # 桶心是画面像素。人站着时，头的中心也是 agent_x 每加 1 就右移 1 像素。
        # 压缩那条线会落在两只脚的左边，左边的桶会被看成在右边。
        self._body_x = 73.0 + (ax - 205)
        interval = max(int(self._since_sample), 1)
        self._since_sample = 0
        # 第一发的出手高度是按打不中的下一帧量的，那时候侧面桶还没稳定，先用帧差看它落到哪。
        # 第一发出去之后只认网络里的侧面桶。墙的抖动和顶上的剩余只数长得不像，不会再进轨迹。
        if getattr(self, "_rule_opened", False):
            if not getattr(self, "_vision_tracks", False):
                self._tracks = []
                self._vision_tracks = True
            from main.barrel_vision import barrel_centers
            points = [(x, y) for x, y, _score in barrel_centers(screen)]
        else:
            points = _barrel_points(prev, screen, rx)
        for track in self._tracks:
            track["pred_x"] = track["x"] + track["vx"] * interval
            track["pred_y"] = track["y"] + track["vy"] * interval
            track["miss"] = track.get("miss", 0) + 1
        used = set()
        for tx, ty in points:
            best_i = None
            best_d = BARREL_TRACK_MATCH
            for i, track in enumerate(self._tracks):
                if i in used:
                    continue
                gap = abs(track["pred_x"] - tx) + abs(track["pred_y"] - ty)
                if gap < best_d:
                    best_d = gap
                    best_i = i
            if best_i is None:
                self._tracks.append({
                    "id": self._next_track,
                    "x": float(tx),
                    "y": float(ty),
                    "vx": 0.0,
                    "vy": 0.0,
                    "samples": 1,
                    "falling": 0,
                    "rolling": 0,
                    "miss": 0,
                    "pred_x": float(tx),
                    "pred_y": float(ty),
                    "fast": 0,
                    "min_y": float(ty),
                    "x_hist": [float(tx)],
                    "step_x": None,
                })
                self._next_track += 1
                continue
            used.add(best_i)
            track = self._tracks[best_i]
            # 只有紧挨着的上一帧才能用来判滚动方向。隔帧补上的位移不算。
            track["step_x"] = (float(tx) - track["x"]) if interval == 1 else None
            track["vx"] = (float(tx) - track["x"]) / interval
            track["vy"] = (float(ty) - track["y"]) / interval
            track["x"] = float(tx)
            track["y"] = float(ty)
            track["min_y"] = min(float(track.get("min_y", ty)), float(ty))
            hist = list(track.get("x_hist") or [])
            hist.append(float(tx))
            track["x_hist"] = hist[-8:]
            track["miss"] = 0
            track["samples"] += 1
            if track["vy"] >= BARREL_FALL_MIN_SPEED:
                track["falling"] += 1
            else:
                # 中间一拍速度不对，不要把已经在掉的桶整段丢掉。
                track["falling"] = max(0, track["falling"] - 1)
            # 连续两拍都是真的在掉，才用这段速度定左右。滑槽口的第一下晃动不算。
            if track["vy"] >= BARREL_DROP_VY:
                track["fast"] = track.get("fast", 0) + 1
            else:
                track["fast"] = 0
            # 连续两拍水平速度同号，而且每帧超过 0.8 像素，方向就定了。
            # 更小的晃动是滑槽口的抖动，不能用来决定左右。
            sign = 1 if track["vx"] > 0.8 else -1 if track["vx"] < -0.8 else 0
            if sign != 0 and sign == track.get("vx_sign", 0) and not track.get("side_lock"):
                track["side_lock"] = sign
                track["lock_y"] = float(track["y"])
            elif sign != 0 and track.get("side_lock") and sign != track["side_lock"]:
                track["side_lock"] = 0
            track["vx_sign"] = sign
            if (
                track["y"] >= BARREL_FLOOR_Y
                and abs(track["vx"]) >= 1.2
                and abs(track["vy"]) < 1.0
            ):
                track["rolling"] = track.get("rolling", 0) + 1
            else:
                track["rolling"] = max(0, track.get("rolling", 0) - 1)
        kept = []
        for i, track in enumerate(self._tracks):
            moving = track.get("falling", 0) >= 1 or track.get("rolling", 0) >= 1
            if i not in used:
                # 这一拍网络没再报这个点，先按上次速度留着。正在掉或正在滚的多留两拍。
                # 这是漏看时的兜底，不拿来判断墙上的桶是不是目标。
                limit = 3 if moving else 1
                if track["miss"] > limit or track["x"] < -30 or track["x"] > 350 or track["y"] > 230:
                    continue
                if track["samples"] > 1 or track["miss"] > 0:
                    track["x"] = track["pred_x"]
                    track["y"] = track["pred_y"]
                track["fast"] = 0
            kept.append(track)
        self._tracks = kept
        self._approach = self._approach_target(rx)

    def _aim_of(self, track, rx):
        """这一只桶该往哪一列走，以及还有多久进入该处理的时候。看不清就返回空。"""
        if track["samples"] < 3:
            return None
        falling = (
            track.get("falling", 0) >= 2
            and track["vy"] >= BARREL_FALL_MIN_SPEED
            and track["y"] < BARREL_LAND_Y
        )
        rolling = (
            track.get("rolling", 0) >= 2
            and track["y"] >= BARREL_FLOOR_Y
            and abs(track["vx"]) >= BARREL_FALL_MIN_SPEED
            and abs(track["vy"]) < 1.0
        )
        if falling:
            # 还在拳头上面，就瞄准它掉到拳头高度的那一列，不等它完全落地。
            if track["y"] < BARREL_STRIKE_Y:
                eta = (BARREL_STRIKE_Y - track["y"]) / track["vy"]
                aim = track["x"] + track["vx"] * eta
            else:
                eta = 0.0
                aim = track["x"]
            if not 0.0 <= aim <= 320.0 or not 0.0 <= eta < 90.0:
                return None
            return "fall", float(aim), float(eta)
        if rolling:
            eta = abs(track["x"] - rx) / max(abs(track["vx"]), 0.2)
            return "roll", float(track["x"]), float(eta)
        return None

    def _approach_target(self, rx):
        """最近一只要处理的桶。空中的瞄准拳头高度，横滚的瞄准它现在的位置。

        人已经在拳头范围内，距离记 0。
        """
        best = None
        for track in self._tracks:
            aim = self._aim_of(track, rx)
            if aim is None:
                continue
            _role, landing, eta = aim
            previous = track.get("landing")
            if previous is None:
                track["landing"] = landing
                track["stable"] = 1
                continue
            # 目标突然跳很远，多半是色块对错了，这一拍不拿来改势。
            if abs(landing - previous) > 40:
                landing = previous
            else:
                landing = 0.6 * previous + 0.4 * landing
                track["stable"] = track.get("stable", 1) + 1
            track["landing"] = landing
            if track.get("stable", 0) < 2:
                continue
            excess = max(0.0, abs(landing - rx) - BARREL_APPROACH_MARGIN)
            # 更急的那只优先。同样急时，选离人更近的。
            rank = (-eta, -excess)
            if best is None or rank > best[0]:
                best = (rank, track["id"], excess)
        if best is None:
            return None
        return best[1], best[2]

    def _state_vector(self):
        """把最多三只桶写成固定长度的数。空位是 0。越急的桶排得越靠前。"""
        rx = float(getattr(self, "_rx", 73.0))
        ranked = []
        for track in self._tracks:
            aim = self._aim_of(track, rx)
            if aim is not None:
                ranked.append((aim[2], aim, track))
        ranked.sort(key=lambda item: item[0])
        vec = np.zeros(BARREL_STATE_DIM, dtype=np.float32)
        for slot, (_eta, aim, track) in enumerate(ranked[:BARREL_STATE_SLOTS]):
            role, aim_x, eta = aim
            base = slot * 8
            vec[base] = 1.0
            vec[base + 1] = float(np.clip((aim_x - rx) / 160.0, -2.0, 2.0))
            vec[base + 2] = float(np.clip(track["y"] / 224.0, 0.0, 1.5))
            vec[base + 3] = float(np.clip(track["vx"] / 4.0, -3.0, 3.0))
            vec[base + 4] = float(np.clip(track["vy"] / 4.0, -3.0, 3.0))
            vec[base + 5] = float(np.clip(eta / 45.0, 0.0, 2.0))
            vec[base + 6] = 1.0 if role == "roll" else 0.0
            vec[base + 7] = 1.0 if aim_x >= rx else -1.0
        vec[-2] = float(np.clip(rx / 320.0, 0.0, 1.2))
        vec[-1] = float(getattr(self, "_facing", 1))
        return vec

    def _rule_threats(self, rx):
        """只留下跟得住的桶。落点突然跳很远的那一拍先不用，避免对着空处出招。"""
        threats = []
        for track in self._tracks:
            if track["samples"] < 4:
                continue
            falling = (
                track.get("falling", 0) >= 3
                and track["vy"] >= BARREL_FALL_MIN_SPEED
                and track["y"] < BARREL_LAND_Y
            )
            rolling = (
                track.get("rolling", 0) >= 3
                and BARREL_PUNCH_Y <= track["y"] <= BARREL_LAND_Y + 4
                and abs(track["vx"]) >= BARREL_FALL_MIN_SPEED
                and abs(track["vy"]) < 1.0
            )
            if falling:
                if track["y"] < BARREL_PUNCH_Y:
                    eta = (BARREL_PUNCH_Y - track["y"]) / track["vy"]
                    aim = track["x"] + track["vx"] * eta
                else:
                    eta = 0.0
                    aim = track["x"]
                if not 0.0 <= aim <= 320.0 or eta > 90:
                    continue
                previous = track.get("rule_aim")
                if previous is not None and abs(aim - previous) > 36:
                    track["rule_aim"] = aim
                    continue
                track["rule_aim"] = aim
                threats.append(("fall", float(aim), float(eta), track))
            elif rolling:
                threats.append(("roll", float(track["x"]), abs(track["x"] - rx) / max(abs(track["vx"]), 0.2), track))
        return threats

    def _travel_side(self, track):
        """桶这几拍净往哪边滑。单帧的抖动会变号，不能拿来定第一发的方向。"""
        hist = list(track.get("x_hist") or [])
        if len(hist) >= 4:
            delta = float(track["x"]) - float(hist[0])
            # 不到 4 像素的来回是色块抖动。桶真在滑时，几帧里会走出这一截。
            if delta <= -4:
                return -1
            if delta >= 4:
                return 1
        locked = int(track.get("side_lock") or 0)
        return locked or None

    def rule_action(self):
        """不经过网络。先走到落桶中间，再朝左站。

        0 等，1 左走，2 右走，3 左轻升龙，4 右轻升龙，5 左波动拳，6 右波动拳。
        先走到落桶中间。出手只看最下面的那只桶：画面上 Y 越大越容易漏过去，更高处的这一拍先不动。
        同一高度里离头近的先打。桶心在 80 到 140 时朝桶升龙，远了也用这一发把身体滑近。
        桶心已经低于 167、横向又超过 45 时，先看后面两帧的滚动方向。
        朝人出升龙，远离出轻拳波动。横向不到 45 不看方向，直接升龙。
        """
        self._rule_cancel = True
        rx = float(getattr(self, "_rx", 73.0))
        # 开局先走到落桶中间，再朝左站一拍把脸转过来，然后才开始打。
        if not getattr(self, "_rule_home", False):
            if rx < BARREL_HOME_X - 4.0:
                return 2
            self._rule_home = True
            return 1

        def dp_side(side):
            return 3 if side < 0 else 4

        # 和桶比左右时用头的中心，不用走到中间的那条压缩坐标。
        body = float(getattr(self, "_body_x", 73.0 + (rx - 73.0) * 1.5))

        def fireball(aim_x):
            return 5 if aim_x < body else 6

        if not getattr(self, "_rule_opened", False):
            self._rule_opened = True
            self._tracks = []
            self._vision_tracks = False
            return 0

        # 最高点击中时桶心在 48。按下之后拳头还要大约 28 帧才到。
        # 桶心还在 34 到 56 就要按，不要等它落到胸口。
        home = 116.0
        pending = getattr(self, "_roll_pending", None)
        if pending is not None:
            # 这两帧只看刚才那只桶往哪边滚，不改去打别的。
            pending["left"] -= 1
            if pending["left"] > 0:
                return 0
            self._roll_pending = None
            track = next(
                (
                    item for item in self._tracks
                    if item.get("id") == pending["id"] and int(item.get("miss", 1)) == 0
                ),
                None,
            )
            if track is not None:
                aim = float(track["x"])
                step = aim - float(pending["x"])
                # 正的差是往右。桶在人左边还往右，或在人右边还往左，就是朝人滚。
                away = (aim < body and step < 0.0) or (aim > body and step > 0.0)
                if away:
                    return fireball(aim)
                if abs(aim - body) < 1.0:
                    return dp_side(-1 if body > home else 1)
                return dp_side(-1 if aim < body else 1)
        barrels = []
        for track in self._tracks:
            x = float(track["x"])
            y = float(track["y"])
            if not (16.0 <= x <= 304.0 and BARREL_APEX_NEAR_LO <= y <= 196.0):
                continue
            if int(track.get("samples", 0)) < 2:
                continue
            # 这一帧网络没再点到的，是按上次速度补出来的位置。补到胸口的空点会引出一发打空的升龙。
            if int(track.get("miss", 0)) > 0:
                continue
            # 圆脸转正时桶心一两帧能下移约 10 像素，仍是同一只桶。一帧跳过 12 像素才是两条轨迹粘住。
            if abs(float(track.get("vy", 0))) > 12.0 or abs(float(track.get("vx", 0))) > 8.0:
                continue
            # 碎块、火花和贴地后几乎不动的圆桶，由识别决定报不报。这里不再按速度把一只桶丢掉。
            barrels.append(track)
        # 先定最下面的那只。Y 越大越靠下，漏过去的风险越高，这一拍只处理它。
        # 和它差 8 像素以内算同一高度，离头近的先打。更高处的桶先放着。
        if barrels:
            lowest = max(float(track["y"]) for track in barrels)
            near = [
                track for track in barrels
                if lowest - float(track["y"]) <= 8.0
            ]
            target = min(near, key=lambda track: abs(float(track["x"]) - body))
            aim = float(target["x"])
            dy = float(target["y"])
            dx = abs(aim - body)
            vy = float(target.get("vy", 0.0))

            def toward():
                # 桶心正好叠在身上，朝回家的方向。
                if abs(aim - body) < 1.0:
                    return dp_side(-1 if body > home else 1)
                return dp_side(-1 if aim < body else 1)

            # 这只还停在头顶附近、没有往下掉，升龙的拳头才能在最高点碰上。
            in_apex = (
                (
                    BARREL_APEX_PRESS_LO <= dy <= BARREL_APEX_PRESS_HI
                    and dx <= BARREL_FIST_RANGE + BARREL_DP_SHIFT + 6.0
                )
                or (
                    BARREL_APEX_NEAR_LO <= dy < BARREL_APEX_PRESS_LO
                    and dx <= BARREL_APEX_NEAR_X
                )
            ) and vy <= BARREL_APEX_HANG_VY
            if in_apex:
                return toward()
            # 已经落到胸口、桶心又叠在身上，升龙打不中。
            on_chest = dy >= 120.0 and dx < 24.0
            # y 大于 167 且横向超过 45，先空过两帧再看滚动方向。不到 45 直接升龙。
            if dy > BARREL_HADOU_Y:
                if dy > BARREL_ROLL_Y and dx > BARREL_FALL_DP_X:
                    self._roll_pending = {"id": target.get("id"), "x": aim, "left": 2}
                    return 0
                if not on_chest:
                    return toward()
            # 桶心在 80 到 140，朝桶出升龙。横向超过 45 时拳头够不到，这一发是让身体朝桶滑近。
            # 46 到 80 这一截如果已经在掉，拳头赶到时桶在胸口，不在这里按。
            if 80.0 <= dy <= BARREL_HADOU_Y and not on_chest:
                return toward()
        # 打桶开始后不再走路。离中间远了，朝中间出升龙，人跟着这下挪回去。
        if abs(body - home) > 14.0:
            return 3 if body > home else 4
        return 0

    def _remember_rule(self, action, score):
        """记下这次攻击。下一步如果分数没涨，就先停一拍。"""
        self._rule_last = action
        self._rule_attack_score = score

    def _decision_observation(self):
        """新模型看位置。旧模型仍看三帧拼出来的运动图。"""
        if getattr(self, "state_obs", False):
            return self._state_vector()
        return self._stack_observation()

    def _approach_shaping(self, before):
        """同一个桶的落点更近才有势。桶消失或换了一只，这一步不加不扣。"""
        after = self._approach
        if before is None or after is None or before[0] != after[0]:
            return 0.0
        phi = -BARREL_APPROACH_SCALE * before[1]
        phi_next = -BARREL_APPROACH_SCALE * after[1]
        shaped = BARREL_SHAPING_GAMMA * phi_next - phi
        return float(np.clip(shaped, -BARREL_SHAPING_CLIP, BARREL_SHAPING_CLIP))

    def _advance_frame(self, frame_action):
        """走一帧模拟器。升龙打空、被打倒这些都在这一帧里结算。"""
        if self.render_enabled:
            self.render_game()
        obs, _, _, _, info = self.env.step(frame_action)
        penalty = self._observe_move(info)
        penalty += self._idle_second_penalty(info)
        status = int(info.get("agent_status", 0))
        if status == BARREL_KNOCKDOWN_STATUS and not self._down:
            penalty += BARREL_KNOCKDOWN_PENALTY
            self._down = True
        elif status != BARREL_KNOCKDOWN_STATUS:
            self._down = False
        if self._boost_frames > 0:
            tick = int(info.get("score", self._boost_from))
            late = max(0, tick - self._boost_from)
            self._boost_from = tick
            if 0 < late <= 4000:
                penalty -= late * BARREL_SCORE_SCALE * BARREL_TURN_BONUS
            self._boost_frames -= 1
        self._since_sample += 1
        self._decision_frames = getattr(self, "_decision_frames", 0) + 1
        # 规则出招时每帧都看。隔 4 帧的话，左边滑槽里闪一下的桶凑不满样本，升龙会跟着右边的背景色块走。
        if self._since_sample >= BARREL_SAMPLE_FRAMES or getattr(self, "_rule_home", False):
            self._sample_tracks(info)
        stage = int(info.get("stage", BONUS_BARREL))
        return obs, info, penalty, stage != BONUS_BARREL

    def _hold_until_landed(self):
        """升龙发出去之后继续空按，直到这下回到站立或确认没打出去。返回多出来的扣分。

        规则也不在半空里交还。打中后的收招里提前按，下一发不会被游戏收，人会站着空等。
        """
        penalty = 0.0
        done = False
        obs = None
        info = self.env.unwrapped.data.lookup_all()
        if self._move is None:
            return obs, info, penalty, done
        for _ in range(BARREL_MOVE_HOLD_FRAMES):
            obs, info, extra, done = self._advance_frame(NO_OP_ACTION.copy())
            penalty += extra
            if done or self._move is None:
                break
        return obs, info, penalty, done

    def step(self, action):
        """走路仍是 4 帧。升龙要等到落地，打中的分和打空都记在这一步。"""
        self.elapsed_steps += 1
        info = self.env.unwrapped.data.lookup_all()
        penalty = 0.0
        screen = self.env.render()
        before = self._approach
        frames = list(action)
        attack = self._attack_pressed(frames)
        done = False
        obs = None
        # 等第一发时一次只走一帧，才能停在打不中的下一帧，而不是一等就是四帧。
        if (
            not attack
            and getattr(self, "_rule_home", False)
            and frames
            and not any(int(v) for frame in frames for v in frame)
        ):
            # 等到拳头能在最高点碰上再出招。一次等四帧会错过这个高度。
            frames = frames[:1]
        if attack and not self._shoryuken_ready(info):
            if getattr(self, "_rule_cancel", False):
                # 打中之后收招比空挥长。人还在 524 里就把下一发按出去，游戏不收，
                # 然后会站着把这下空指令耗完。等到回到站立再按，才是接得上的最快一拍。
                waited = 0
                while not done and not self._shoryuken_ready(info) and waited < 90:
                    obs, info, extra, done = self._advance_frame(NO_OP_ACTION.copy())
                    penalty += extra
                    waited += 1
                if done or not self._shoryuken_ready(info):
                    frames = []
                    attack = False
                else:
                    # 倒地或收招里定的方向，站稳时桶已经不在那边。站稳再看一次。
                    fresh = int(self.rule_action())
                    if fresh >= 3:
                        frames = [
                            np.array(frame, copy=True)
                            for frame in barrel_action_sequences()[fresh]
                        ]
                    else:
                        frames = []
                        attack = False
            elif self._can_cancel_into(info):
                # 试验过这一帧四种接招都出得来。先结掉上一发，再把这一发送出去。
                self._handed_frames = int(self._move.get("frames", 0))
                self._handed_intend = self._move.get("intend")
                self._move = None
            else:
                # 收招没完就再按，方向键会接成波动拳，拳键会变成轻拳。这一下直接不送。
                frames = [NO_OP_ACTION.copy() for _ in range(len(frames))]
                penalty += BARREL_BLOCKED_PENALTY
                attack = False
        if attack and self._move is None and frames:
            forward = self._dp_forward(frames)
            # 升龙和波动都用轻拳。波动的第一帧只有下，升龙的第一帧是前，靠这个分开。
            starts_down = (
                int(frames[0][BUTTON_DOWN])
                and not int(frames[0][BUTTON_LEFT])
                and not int(frames[0][BUTTON_RIGHT])
            )
            intend = "hadou" if starts_down else "dp"
            name = "LEFT" if forward == BUTTON_LEFT else "RIGHT"
            want = -1 if forward == BUTTON_LEFT else 1
            already = int(getattr(self, "_facing", 1)) == want
            if intend == "dp":
                # 反向先朝目标按两帧，脸转过来，再前、下、下前、下前加轻拳。只按一帧前仍是轻拳。
                frames = shoryuken_frames(name, turn_first=not already)
            if intend == "hadou":
                frames = hadouken_frames(name, turn_first=not already)
                # 上一发波动拳刚能接升龙时，马上再搓波动拳会被收成轻拳。空到中拳落在接得上的那一帧。
                if (
                    getattr(self, "_rule_cancel", False)
                    and getattr(self, "_handed_intend", None) == "hadou"
                ):
                    pad = BARREL_HADOU_CHAIN_PUNCH - (int(self._handed_frames) + len(frames) - 1)
                    for _ in range(max(0, pad)):
                        obs, info, extra, done = self._advance_frame(NO_OP_ACTION.copy())
                        penalty += extra
                        if done:
                            frames = []
                            break
            # 旧的身前色块规则对不准会不会打中，这里的扣分是 0，不靠它拦招。
            if frames and not self._shoryuken_aimed(info, forward, screen):
                penalty += BARREL_UNAIMED_PENALTY
            if frames:
                self._begin_move(info, forward, intend)
        elif frames and int(frames[0][BUTTON_LEFT]) and not int(frames[0][BUTTON_RIGHT]):
            self._facing = -1
        elif frames and int(frames[0][BUTTON_RIGHT]) and not int(frames[0][BUTTON_LEFT]):
            self._facing = 1
        # 存这一手开始前的画面。下一步用它和最新画面相减，桶的移动才看得出来。
        self._aim_prev = None if screen is None else np.array(screen, copy=True)
        self._decision_frames = 0
        for frame_action in frames:
            obs, info, extra, done = self._advance_frame(frame_action)
            penalty += extra
            if done:
                break
        landed = False
        if not done and self._move is not None:
            held_obs, info, extra, done = self._hold_until_landed()
            penalty += extra
            if held_obs is not None:
                obs = held_obs
            landed = self._move is None
        elif attack and not done:
            landed = True
        # 规则在能接下一招时就交还控制，不再空等分数跳完。训练仍等这一下落地。
        if landed and not done and not getattr(self, "_rule_cancel", False):
            # 升龙落地后分数有时晚几帧才跳。波动拳的火球更晚，多等一会儿，仍算这一下。
            settle_for = BARREL_SCORE_SETTLE_FRAMES
            if getattr(self, "_last_intend", None) == "hadou":
                settle_for = BARREL_HADOU_SETTLE_FRAMES
            for _ in range(settle_for):
                obs, info, extra, done = self._advance_frame(NO_OP_ACTION.copy())
                penalty += extra
                if done:
                    break
        if not getattr(self, "state_obs", False):
            if obs is None:
                obs = self.env.render()
            obs = self._process_frame(obs)
            self.frame_stack.append(obs)
        if "score" not in info or "stage" not in info:
            info = self.env.unwrapped.data.lookup_all()
        score = int(info.get("score", self._prev_score))
        gained_now = max(0, score - self._prev_score)
        self._prev_score = score
        # 波动拳先挂着的打空，这一步里分数涨了就取消。
        if gained_now > 0:
            self._pending_whiff = 0.0
        penalty += float(getattr(self, "_pending_whiff", 0.0) or 0.0)
        self._pending_whiff = 0.0
        shaping = self._approach_shaping(before)
        reward = gained_now * BARREL_SCORE_SCALE - self.time_penalty - penalty + shaping
        stage = int(info.get("stage", BONUS_BARREL))
        done = stage != BONUS_BARREL
        truncated = (not done) and self.elapsed_steps >= self.max_episode_steps
        info["barrel_gained"] = score - self._score_at_reset
        info["barrel_state"] = getattr(self, "_barrel_state_name", "")
        info["barrel_shaping"] = shaping
        info["barrel_frames"] = int(getattr(self, "_decision_frames", 0))
        return self._decision_observation(), float(reward), done, truncated, info


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


def shoryuken_frames(forward, turn_first):
    """轻拳升龙。同向是前、下、下前、下前加轻拳。

    反向时脸还没转过来，这一条会被收成轻拳。先朝目标按两帧把脸转正，再出同样四拍。
    只先按一帧不够，人还是挥轻拳。
    """
    buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
               'C', 'Y', 'X', 'Z')

    def frame(*pressed):
        return [int(button in pressed) for button in buttons]

    motion = [
        frame(forward),
        frame('DOWN'),
        frame('DOWN', forward),
        frame('DOWN', forward, 'X'),
    ]
    if not turn_first:
        return motion
    return [frame(forward), frame(forward), *motion]


def hadouken_frames(forward, turn_first):
    """轻拳波动拳。已经朝这边时不能先按方向，否则会被收成升龙。

    要转身时先朝目标按两帧，再空两帧，让方向键离开升龙的指令窗口，然后才下、下前、前、前加轻拳。
    """
    buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
               'C', 'Y', 'X', 'Z')

    def frame(*pressed):
        return [int(button in pressed) for button in buttons]

    motion = [
        frame('DOWN'),
        frame('DOWN', forward),
        frame(forward),
        frame(forward, 'X'),
    ]
    if not turn_first:
        return motion
    return [frame(forward), frame(forward), frame(), frame(), *motion]


def barrel_action_sequences():
    """木桶的动作。等、左走、右走、左轻升龙、右轻升龙、左轻拳波动、右轻拳波动。

    看准之后轻升龙就够。波动只留给贴地后正在远离、横向又超过 45 的那一只。
    脸还朝另一边时，先朝目标方向站两帧把脸转过来，再出招。
    """
    buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
               'C', 'Y', 'X', 'Z')

    def frame(*pressed):
        return [int(button in pressed) for button in buttons]

    def shoryuken(forward):
        # 轻拳是 X。面朝左和面朝右用的前方向相反，所以左右两套指令不同。
        return [
            frame(forward),
            frame(forward),
            frame(forward),
            frame('DOWN'),
            frame('DOWN', forward),
            frame('DOWN', forward, 'X'),
        ]

    def hadouken(forward):
        # 轻拳是 X。第一帧只有下，送进模拟器前靠这一点和升龙分开。
        return hadouken_frames(forward, turn_first=False)

    return [
        [frame()] * 4,
        [frame('LEFT')] * 4,
        [frame('RIGHT')] * 4,
        shoryuken('LEFT'),
        shoryuken('RIGHT'),
        hadouken('LEFT'),
        hadouken('RIGHT'),
    ]


class BarrelActionWrapper(gym.ActionWrapper):
    """木桶的出招：等、左右走、左右轻拳升龙，以及左右轻拳波动拳。

    连续闯关进木桶时用的就是这七个动作，和三份存档同一套规则。
    """

    def __init__(self, env):
        super().__init__(env)
        self._actions = barrel_action_sequences()
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]
