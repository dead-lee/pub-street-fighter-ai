"""拆文件之后，出招和分数仍要和拆开前一样。"""

import gymnasium as gym
import numpy as np

from main.barrel_play import (
    BARREL_BODY_HOME,
    BARREL_FAR_HADOU_HI,
    BARREL_Y_LO,
    BarrelActionWrapper,
    BarrelBonusWrapper,
    barrel_action_sequences,
)
from main.bonus import BonusRoutine
from main.buttons import BUTTON_LEFT, BUTTON_RIGHT, BUTTON_X
from main.fight import SFighterActionWrapper
from main.saves import barrel_state_paths
from main.score_skip import result_skip_action


class _Dummy(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.MultiBinary(12)
        self.observation_space = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)


def test_fight_actions_stay_twenty_seven():
    wrapper = SFighterActionWrapper(_Dummy())
    assert wrapper.action_space.n == 27
    assert wrapper.action(0) == [[0] * 12] * 4


def test_barrel_actions_stay_seven_light_moves():
    actions = barrel_action_sequences()
    assert len(actions) == 7
    assert BarrelActionWrapper(_Dummy()).action_space.n == 7
    # 右轻升龙的最后一拍只按轻拳，不带方向。轻拳是 X，下标 10。
    assert actions[4][-1][10] == 1
    assert actions[4][-1][5] == 0
    assert actions[4][-1][7] == 0
    assert actions[6][-1][10] == 1
    assert actions[6][-1][7] == 0
    # 最后一拍没有方向。朝向要从出拳前的方向键读，左波动不能落到右边。
    gate = BarrelBonusWrapper.__new__(BarrelBonusWrapper)
    assert gate._dp_forward(actions[3]) == BUTTON_LEFT
    assert gate._dp_forward(actions[4]) == BUTTON_RIGHT
    assert gate._dp_forward(actions[5]) == BUTTON_LEFT
    assert gate._dp_forward(actions[6]) == BUTTON_RIGHT


def test_score_skip_waits_one_second():
    assert result_skip_action(59, False)[BUTTON_X] == 0
    assert result_skip_action(60, False)[BUTTON_X] == 1
    assert result_skip_action(60, True)[BUTTON_X] == 0


def test_live_saves_keep_their_order():
    names = [path.split("/")[-1] for path in barrel_state_paths(include_original=False)]
    assert names == [
        "Champion.X.BonusBarrel.Live01.state",
        "Champion.X.BonusBarrel.Live02.state",
        "Champion.X.BonusBarrel.Live03.state",
    ]


def test_barrel_rule_keeps_the_same_choices():
    player = BarrelBonusWrapper.__new__(BarrelBonusWrapper)
    player._rule_home = False
    player._rule_opened = False
    player._rx = 73.0
    player._body_x = 73.0
    player._tracks = []
    player._roll_pending = None
    # 身体还没到中线，开局往右走。
    assert player.rule_action() == 2
    player._body_x = BARREL_BODY_HOME - 1
    assert player.rule_action() == 2
    # 走到中线就停。这一拍清掉跟踪并等待，下一拍没有候选桶也等待。
    player._body_x = BARREL_BODY_HOME
    assert player.rule_action() == 0
    assert player.rule_action() == 0
    player._body_x = 116.0
    player._roll_pending = None
    # Y=100、dx=20，落在出招表第 3 行，朝桶升龙。
    player._tracks = [{
        "id": 1, "x": 136.0, "y": 100.0, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    assert player.rule_action() == 4
    # Y=100、dx=84，落在高位到脚边之间的远桶一行，朝桶波动。
    player._tracks = [{
        "id": 4, "x": 200.0, "y": 100.0, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 6
    # Y=34 低于 50，不进候选。人在 home，等待。
    player._tracks = [{
        "id": 6, "x": 136.0, "y": 34.0, "samples": 4, "miss": 0, "vy": 20.0, "vx": 20.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 0
    # 比 Y 下限低 1，近桶也不进候选。人在 home，等待。
    player._tracks = [{
        "id": 7, "x": 136.0, "y": BARREL_Y_LO - 1, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 0
    # 刚好到 Y 下限、dx=84，高处的远桶，朝桶升龙。速度再大也进候选。
    player._tracks = [{
        "id": 5, "x": 200.0, "y": BARREL_Y_LO, "samples": 4, "miss": 0, "vy": 20.0, "vx": 20.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 4
    # 脚边减 1 还在波动拳那一行。到了脚边才开始看滚动。
    player._tracks = [{
        "id": 8, "x": 200.0, "y": BARREL_FAR_HADOU_HI - 1, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 6
    # Y=170、dx 很远。A、B、C 连续往左，两段同向，出左波动。
    player._tracks = [{
        "id": 2, "x": 24.0, "y": 170.0, "samples": 4, "miss": 0, "vy": 0.2, "vx": -0.5,
    }]
    player._roll_pending = None
    assert player.rule_action() == 0
    player._tracks[0]["x"] = 22.0
    assert player.rule_action() == 0
    player._tracks[0]["x"] = 20.0
    assert player.rule_action() == 5
    # 前两段一右一左，方向不同，再等一帧。后两段都往左，出左波动。
    player._tracks = [{
        "id": 9, "x": 30.0, "y": 170.0, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 0
    player._tracks[0]["x"] = 32.0
    assert player.rule_action() == 0
    player._tracks[0]["x"] = 30.0
    assert player.rule_action() == 0
    player._tracks[0]["x"] = 28.0
    assert player.rule_action() == 5
    # 连着四段都对不上。第五帧用 E 减 A，不看最后一小段。
    # A=40，E=30，整段往左。最后一小段 20→30 是往右，若用它会出升龙。
    player._tracks = [{
        "id": 10, "x": 40.0, "y": 170.0, "samples": 4, "miss": 0, "vy": 0.0, "vx": 0.0,
    }]
    player._roll_pending = None
    assert player.rule_action() == 0
    for x in (20.0, 40.0, 20.0):
        player._tracks[0]["x"] = x
        assert player.rule_action() == 0
    player._tracks[0]["x"] = 30.0
    assert player.rule_action() == 5


def test_car_waits_until_ryu_is_on_stage():
    action = BonusRoutine().action(12, {"agent_hp": 0, "agent_x": 0, "score": 0})
    assert action.tolist() == [0] * 12
