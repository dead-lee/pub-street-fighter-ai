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


"""12 关和滚木桶的开局存档。"""

import os

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
    prefix = "Champion.X.BonusBarrel.Live"
    for name in sorted(os.listdir(DATA_DIR)):
        if name.startswith(prefix) and name.endswith(".state"):
            paths.append(os.path.join(DATA_DIR, name))
    if not paths:
        paths.append(BARREL_STATE_PATH)
    return paths
