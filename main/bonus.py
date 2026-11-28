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


"""砸汽车和打砖墙的固定按键。滚木桶不走这里。"""

import numpy as np
from main.buttons import (
    BUTTON_C,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP,
    BUTTON_Y,
    BUTTON_Z,
    _blank_action,
)
from main.stages import BONUS_CAR, BONUS_WALL

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
