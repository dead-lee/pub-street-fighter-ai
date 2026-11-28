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


"""滚木桶的加分、扣分和出招。

对战模型不看这一关。连续闯关和三份存档用的是同一套规则。
"""

import gymnasium as gym
import cv2
import collections
import random
import os
from pathlib import Path

import numpy as np
import openpyxl
from main.buttons import (
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_X,
    BUTTON_Y,
    NO_OP_ACTION,
)
from main.fight import SFighterCustomWrapper
from main.saves import barrel_state_paths
from main.stages import BONUS_BARREL

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


def _roll_delta(samples):
    """滑动窗口看桶往哪边滚。返回这一拍要用的 ΔX。还不能定方向时返回空。

    人站稳后从第一帧开始记 X。不满三帧只排队。满了就比最后两段：
    两段都往左或都往右，用最新这一段。一段没动（ΔX 为 0）不算有方向。
    两段方向不同就再等一帧，用新的两段再比。记满 A 到 E 仍不同向，用 E 减 A。
    """
    if len(samples) < 3:
        return None
    # 五帧对不齐时，不看最后一小段，用整段的两端。
    if len(samples) >= BARREL_ROLL_FRAMES:
        return samples[-1] - samples[0]
    earlier = samples[-2] - samples[-3]
    latest = samples[-1] - samples[-2]
    same_way = (earlier > 0.0 and latest > 0.0) or (earlier < 0.0 and latest < 0.0)
    if same_way:
        return latest
    return None


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
# 出招表的上下限。数值在「木桶出招.xlsx」的「配置」里，表上的区间公式也指向同一格。
def _load_rule_limits():
    """读配置页。改表里的数之后，下一局代码和区间显示一起变。"""
    path = Path(__file__).resolve().parents[1] / "木桶出招.xlsx"
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        found = {}
        for key, value, *_rest in book["配置"].iter_rows(min_row=3, max_col=2, values_only=True):
            if key:
                found[str(key)] = value
    finally:
        book.close()
    missing = [name for name in ("Y下限", "Y上限", "X下限", "X上限", "近距", "高位", "脚边", "样本", "同高", "叠身", "中线", "离中线", "滚动帧", "落点") if name not in found]
    if missing:
        raise RuntimeError(f"木桶出招.xlsx 的配置缺了 {missing}")
    return found


_LIMITS = _load_rule_limits()
# 候选要同时落在这些范围里。低于 Y 下限的桶还在头顶，这一拍不因它出招。速度不拦候选。
BARREL_Y_LO = float(_LIMITS["Y下限"])
BARREL_Y_HI = float(_LIMITS["Y上限"])
BARREL_X_LO = float(_LIMITS["X下限"])
BARREL_X_HI = float(_LIMITS["X上限"])
# dx 不超过近距，朝桶升龙。等于近距也走这一行。
BARREL_CLOSE_DX = float(_LIMITS["近距"])
# dx 超过近距时按高度分。低于高位朝桶升龙，把身体带近。
# 高位到脚边之间提前朝桶出波动拳。脚边起滑动看滚动方向。
BARREL_FAR_HADOU_LO = float(_LIMITS["高位"])
BARREL_FAR_HADOU_HI = float(_LIMITS["脚边"])
BARREL_MIN_SAMPLES = int(_LIMITS["样本"])
# 和最下面那只相差在这个数以内，才比较谁离 body 近。
BARREL_SAME_BAND = float(_LIMITS["同高"])
# 桶心离 body 小于这个数，升龙改朝中线。
BARREL_ON_BODY = float(_LIMITS["叠身"])
# 贴地滚动最多看这么多帧。相邻两段同向就出招，记满仍不同向就用最后一帧减第一帧。
BARREL_ROLL_FRAMES = int(_LIMITS["滚动帧"])
# 桶心叠在身上时，升龙朝这里。没有候选桶时不再因为离中线而补位。
BARREL_BODY_HOME = float(_LIMITS["中线"])
BARREL_HOME_SLACK = float(_LIMITS["离中线"])
# 落点是压缩坐标。开局按身体中心走到中线，不用这个数。
BARREL_HOME_X = float(_LIMITS["落点"])
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
        # 开局先走到中线。走到之后被撞开也不再退回只走路。
        self._rule_home = False
        self._rule_opened = False
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
        """朝向在出拳之前的方向键上。最后一拍只有拳，那一拍不再表示左右。

        升龙的前在第一拍，波动的前在下之后。先出现的那个左右就是这一发的朝向。
        """
        for frame in action:
            if int(frame[BUTTON_X]) or int(frame[BUTTON_Y]):
                continue
            if int(frame[BUTTON_LEFT]):
                return BUTTON_LEFT
            if int(frame[BUTTON_RIGHT]):
                return BUTTON_RIGHT
        for frame in action:
            if int(frame[BUTTON_LEFT]):
                return BUTTON_LEFT
            if int(frame[BUTTON_RIGHT]):
                return BUTTON_RIGHT
        return BUTTON_RIGHT

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
                })
                self._next_track += 1
                continue
            used.add(best_i)
            track = self._tracks[best_i]
            track["vx"] = (float(tx) - track["x"]) / interval
            track["vy"] = (float(ty) - track["y"]) / interval
            track["x"] = float(tx)
            track["y"] = float(ty)
            track["miss"] = 0
            track["samples"] += 1
            if track["vy"] >= BARREL_FALL_MIN_SPEED:
                track["falling"] += 1
            else:
                # 中间一拍速度不对，不要把已经在掉的桶整段丢掉。
                track["falling"] = max(0, track["falling"] - 1)
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

    def rule_action(self):
        """不经过网络。开局先走到中线，再按出招表打。

        0 等，1 左走，2 右走，3 左轻升龙，4 右轻升龙，5 左波动拳，6 右波动拳。
        出招表从上到下，先成立的一行出招。空着的列不看。
        先取 Y 最大的那只。和它相差在同高以内的里面，留离 body 最近的一只。
        """
        self._rule_cancel = True
        rx = float(getattr(self, "_rx", 73.0))

        def dp_side(side):
            return 3 if side < 0 else 4

        # 和桶比左右时用身体中心，不用走到中间的那条压缩坐标。
        body = float(getattr(self, "_body_x", 73.0 + (rx - 73.0) * 1.5))

        def dp_home():
            # body 不大于 home 时朝右。已经站在中线上，左右都朝 home。
            return dp_side(-1 if body > BARREL_BODY_HOME else 1)

        def dp_toward(aim_x):
            # 桶心叠在身上时不按桶的左右，朝 home。
            if abs(aim_x - body) < BARREL_ON_BODY:
                return dp_home()
            return dp_side(-1 if aim_x < body else 1)

        def hadou_toward(aim_x):
            return 5 if aim_x < body else 6

        # 走到中线之前不看出招表。右走一拍大约 7 像素，109 的下一拍正好落在 116。
        if not getattr(self, "_rule_home", False):
            if body < BARREL_BODY_HOME:
                return 2
            self._rule_home = True

        if not getattr(self, "_rule_opened", False):
            self._rule_opened = True
            self._tracks = []
            self._vision_tracks = False
            return 0

        pending = getattr(self, "_roll_pending", None)
        if pending is not None:
            track = next(
                (item for item in self._tracks if item.get("id") == pending["id"]),
                None,
            )
            if track is not None:
                # 窗口进行中不再看 Y。每拍补上现在的 X，同向才出招。
                pending["xs"].append(float(track["x"]))
                step = _roll_delta(pending["xs"])
                if step is None:
                    return 0
                self._roll_pending = None
                aim = float(track["x"])
                if aim < body and step < 0.0:
                    return 5
                if aim > body and step > 0.0:
                    return 6
                if abs(aim - body) < BARREL_ON_BODY:
                    return dp_home()
                return dp_toward(aim)
            # 这只桶已经不在，按当前最下面的桶重判。
            self._roll_pending = None
        barrels = []
        for track in self._tracks:
            x = float(track["x"])
            y = float(track["y"])
            # X、Y 超出配置的上下限就不进候选。
            if not (BARREL_X_LO <= x <= BARREL_X_HI and BARREL_Y_LO <= y <= BARREL_Y_HI):
                continue
            if int(track.get("samples", 0)) < BARREL_MIN_SAMPLES:
                continue
            barrels.append(track)
        if barrels:
            lowest = max(float(track["y"]) for track in barrels)
            near = [
                track for track in barrels
                if lowest - float(track["y"]) <= BARREL_SAME_BAND
            ]
            target = min(near, key=lambda track: abs(float(track["x"]) - body))
            aim = float(target["x"])
            dy = float(target["y"])
            dx = abs(aim - body)
            # dx 不超过近距，不论高度，朝桶升龙。等于近距也走这一行，不再往下看。
            if dx <= BARREL_CLOSE_DX:
                return dp_toward(aim)
            # 高处的远桶，朝桶升龙，把身体带近。
            if dy < BARREL_FAR_HADOU_LO:
                return dp_toward(aim)
            # 中等高度的远桶，提前朝桶出波动拳。脚边那个高度不含在这一行里。
            if dy < BARREL_FAR_HADOU_HI:
                return hadou_toward(aim)
            # 落到脚边的远桶。这一拍是窗口的第一帧，先记下 X。
            self._roll_pending = {"id": target.get("id"), "xs": [aim]}
            return 0
        # 没有候选桶就等待。不再因为离中线而朝中线升龙。
        return 0

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
        # 等待只走一帧。一次走四帧，桶会跨过出招表的一行。
        if (
            not attack
            and getattr(self, "_rule_home", False)
            and frames
            and not any(int(v) for frame in frames for v in frame)
        ):
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
                # 反向先朝目标按两帧，脸转过来，再前、下、下前、拳。最后一拍不带方向。
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

def shoryuken_frames(forward, turn_first):
    """轻拳升龙。同向是前、下、下前、拳。

    最后一拍只按拳，方向键先松开。下一发再按前时，不会和这一拍的下前接成波动拳。
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
        frame('X'),
    ]
    if not turn_first:
        return motion
    return [frame(forward), frame(forward), *motion]


def hadouken_frames(forward, turn_first):
    """轻拳波动拳。已经朝这边时不能先按方向，否则会被收成升龙。

    要转身时先朝目标按两帧，再空两帧，让方向键离开升龙的指令窗口，然后才下、下前、前、拳。
    最后一拍只按拳，不把方向留在指令里。
    """
    buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
               'C', 'Y', 'X', 'Z')

    def frame(*pressed):
        return [int(button in pressed) for button in buttons]

    motion = [
        frame('DOWN'),
        frame('DOWN', forward),
        frame(forward),
        frame('X'),
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
        # 轻拳是 X。最后一拍只按拳。面朝左和面朝右用的前方向相反。
        return [
            frame(forward),
            frame(forward),
            frame(forward),
            frame('DOWN'),
            frame('DOWN', forward),
            frame('X'),
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
