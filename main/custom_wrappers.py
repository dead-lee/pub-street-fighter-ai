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


"""原来的总入口。画面、奖励关和木桶已经按职责拆开，这里只留下旧名字。

对战在 fight.py，砸汽车和打砖墙在 bonus.py，木桶在 barrel_play.py。
"""

from main.barrel_play import *  # noqa: F401,F403
from main.bonus import *  # noqa: F401,F403
from main.buttons import *  # noqa: F401,F403
from main.fight import *  # noqa: F401,F403
from main.saves import *  # noqa: F401,F403
from main.score_skip import *  # noqa: F401,F403
from main.stages import *  # noqa: F401,F403
