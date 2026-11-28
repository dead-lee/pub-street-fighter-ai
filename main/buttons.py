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


"""手柄 12 个键的下标。顺序和模拟器里的 MD 手柄一致。"""

import numpy as np

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

def _blank_action():
    return np.zeros(12, dtype=np.int8)
