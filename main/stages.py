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


"""街机插进来的三关奖励。编号来自 data.json 里的 stage。"""

# data.json 里的 stage。0–11 是对战场地，这三关是街机插进来的 bonus。
BONUS_CAR = 12
BONUS_WALL = 13
BONUS_BARREL = 14
BONUS_STAGE_NAMES = {
    BONUS_CAR: "砸汽车",
    BONUS_WALL: "打砖墙",
    BONUS_BARREL: "滚木桶",
}
