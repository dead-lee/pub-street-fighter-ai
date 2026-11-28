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


"""结算画面上跳过后面的跳分。"""

import numpy as np

from main.buttons import BUTTON_X

# 分数开始一格格累加之后，先看 1 秒，再点一次轻拳把剩下的一次加上。
# 胜利姿势刚出来时按键不会跳过结算，所以要等分数真的开始跳。
RESULT_HOLD_FRAMES = 60
RESULT_TAP_FRAMES = 3
# 正常是一小截一小截地加。一次加上八千以上，就是跳过已经生效。
RESULT_SKIP_SCORE_JUMP = 8000

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
