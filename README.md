# Ultimate SFighterAI

本项目基于深度强化学习训练了一个能够打通《街头霸王 2：冠军特别版》（Street Fighter II: Special Champion Edition）全部 12 个街机关卡的卷积神经网络模型（NatureCNN）。我们使用隆（Ryu）作为玩家角色，完全基于游戏画面（RGB 像素值）进行决策，通过 PPO 算法学习移动、攻击和释放波动拳、旋风腿、升龙拳等招式。

本版本延续了[旧版项目](https://github.com/linyiLYi/street-fighter-ai/blob/master/README_CN.md)的训练思路，将训练范围从关底 BOSS 扩展至全部 12 个关卡。各关卡在独立的模拟器中并行运行，共同训练同一个 NatureCNN 模型。项目提供了终版模型权重和对应的训练记录，可以复现训练，或是运行测试观看 AI 模型的对战表现。

> 模型输入仅包含游戏画面。训练环境会读取内存变量中的敌我双方血量 `agent_hp` 和 `enemy_hp`，用于计算奖励和判断胜负。

### 文件结构

```text
.
├── data/                        # 关卡存档与游戏配置
│   ├── Champion.X.Level*.state  # 第 1～12 关开局存档，难度为菜单 8 星
│   ├── Champion.X.Bonus*.state  # 砸汽车、打砖墙、滚木桶的开打存档
│   ├── data.json                # 游戏内存变量定义，含关卡编号 stage
│   └── scenario_full_frame.json # 游戏场景配置
├── main/                        # 训练与测试代码
│   ├── train.py                 # 全关卡并行训练与续训
│   ├── train_barrel.py          # 只训练滚木桶，出招锁定轻拳升龙
│   ├── test.py                  # 逐关测试、连续闯关、全屏原速演示
│   ├── presenter.py             # 全屏画面和声音
│   ├── env_utils.py             # ROM 配置与环境构建
│   └── custom_wrappers.py       # 画面处理、动作映射、奖励与奖励关套路
├── logs/
│   ├── trained_models/
│   │   ├── ppo_ryu.zip          # 对战模型权重
│   │   └── ppo_barrel_ryu.zip   # 滚木桶模型权重，连续闯关走到木桶时使用
│   └── events.out.tfevents.*    # 训练记录
├── requirements.txt
├── LICENSE
└── README.md
```

游戏配置文件存储在 `data/` 文件夹下；项目的主要代码文件夹为 `main/`。`logs/` 中包含了训练过程的数据曲线，`logs/trained_models/` 中包含了模型权重文件，可以用于在 `test.py` 中运行测试，观看 AI 模型学习到的对战策略。

## 运行指南

本项目基于 Python 编程语言，主要使用了 [Gymnasium](https://gymnasium.farama.org/)、[Stable-Retro](https://github.com/Farama-Foundation/stable-retro)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 等开源代码库。Python 最低版本是 3.10，3.10 及以上都可以，包括 3.14。`requirements.txt` 里的版本号同样是最低要求，更高的兼容版本可以用。当前已在 macOS / Apple Silicon 上用 Python 3.14 跑通 CPU 和 Apple MPS。以下为控制台/终端指令，均从项目根目录执行。

### 环境配置

```bash
# 用 Python 3.10 或更高版本创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

也可以使用已有的 [Anaconda](https://www.anaconda.com/) Python 3.10 及以上环境，再安装 `requirements.txt` 中的依赖。Windows 使用虚拟环境时，激活命令为 `.venv\Scripts\activate`；其他平台的安装要求请参考 Stable-Retro 和 PyTorch 的官方说明。

运行程序还需要《街头霸王·二：冠军特别版》美版的游戏 ROM 文件（可以理解为游戏程序本身）。Stable-Retro 不提供游戏 ROM，本项目也不包含该文件，需要自行通过合法途径获得。准备好支持的 ROM 后，通过环境变量指定其绝对路径：

```bash
export SF2_ROM_PATH="/absolute/path/to/your/rom.md"
```

Windows PowerShell 中使用 `$env:SF2_ROM_PATH = "C:\path\to\your\rom.md"`。也可以将 ROM 放在项目的 `data/street-fighter-ii-special-champion-edition-usa.md` 路径下。训练和测试程序会自动导入 ROM；如果当前 Python 环境的 Stable-Retro 已导入该 ROM，可以直接使用。

`data/` 中的 12 个 `.state` 文件保存了各关卡的开局状态，两个 `.json` 文件定义了游戏内存变量和场景配置。程序会直接读取这些文件，无需手动复制到 Stable-Retro 的安装目录。至此，环境配置准备工作完成。

### 运行测试

环境配置完成后，可以运行 `main/test.py`，实际体验智能代理的对战表现：

```bash
python -m main.test --episodes 1
```

默认测试会依次加载第 1～12 关的开局存档，每关采用三局两胜。`--episodes 1` 表示完整测试一轮全部 12 关对战；省略该参数时，默认测试 10 轮。某关失败后仍会继续测试后面的关卡，终端会输出每关的胜负、奖励、关卡胜率，以及十二关全部获胜的比例。

如果只关心测试结果、不需要观看对局，可以关闭游戏画面：

```bash
python -m main.test --episodes 1 --no-render --device cpu
```

如果想观看从第 1 关开始的连续闯关，可以运行：

```bash
python -m main.test --episodes 1 --no-load-level-saves
```

连续闯关只在开局加载第一关存档，之后通过游戏自身的关卡推进。演示不会在 12 场后自己结束，木桶、四大天王和结局都会继续播，直到关掉窗口。

全屏、原速、有声音的演示：

```bash
python -m main.test --episodes 1 --no-load-level-saves --realtime
```

窗口按游戏原速播放，画面拉大后仍是一块块像素。画面铺满屏幕，但不是独占全屏，系统快捷键不会被吃掉。程序不会自己关窗口，看完后按 Esc 或关闭窗口退出。结算分数开始跳动 1 秒后，会点一次轻拳把后面的跳分跳过，胜利姿势、台词和世界地图仍会播放。

### 奖励关

连续闯关走到奖励关时，不把画面交给对战模型，而用固定按键。对战模型没见过这些画面。

| 关卡 | 当前结果 |
| --- | --- |
| 砸汽车 | 左边贴身一记重脚（这一脚打两段），再重拳升龙翻到右边，上半截重踢、底盘蹲中拳。可以 PERFECT |
| 打砖墙 | 重拳升龙上到左侧高处后，先把三脚有伤害的重腿踢完，再走到右边地面踢三脚二段。贴最后一段用朝砖跳，不用升龙或旋风腿。存档实测第 1915 帧拆完，得分 544300 |
| 滚木桶 | 连续闯关使用 `logs/trained_models/ppo_barrel_ryu.zip`。存档实测加分 62300，左右升龙都会出。固定套路仍是 17/20、加分 30300，只作对照 |

木桶的落点会随出手时机变化。`main/train_barrel.py` 用和对战相同的 PPO、84×84 运动图和 NatureCNN，只在木桶存档上训练。动作只有五个：站着等、往左走、往右走、朝左轻拳升龙、朝右轻拳升龙。走路按住 4 帧。升龙是 6 帧：先朝目标方向站两帧把脸转过来，再接前、下、下前、下前加轻拳。面朝右时直接按朝左的键，游戏会收成轻拳或波动拳。

奖励按游戏实际加上的分数计，每一千分给 10。升龙打空扣 1，打成轻拳或波动拳扣 2。人还没收招就再按，这一下被吃掉。从第 2 秒起，一整秒没打出伤害再扣 1。和上一发升龙方向相反而且打中时，这一下的分数再给一倍。当前权重是 150 万步时评估最好的一版，三局都是 +62300。连续闯关走到木桶就用它。单独看这一局：

```bash
python -m main.train_barrel --eval-only logs/trained_models/ppo_barrel_ryu.zip --eval-episodes 1 --watch
```

从头训练：

```bash
python -m main.train_barrel --num-envs 8 --total-timesteps 2000000
```

这几轮训练里踩过的点：

- 打中给的分如果盖不过空挥、串招和罚站，策略会改成站着不动。早期每一千分只给 1 时就是这样。把打中提高到每一千分给 10 之后，出手才比干站更赚。
- 只重罚串招、取消空挥之后，乱按一局的扣分比站着还多，策略同样停手。空挥和串招都要扣，但扣分不能压过打中。
- 换边加分只有在相反方向真的升起来时才进得了训练数据。指令没把脸转过来时，朝左的键不是升龙，加到三成或一倍都不会被抽到。
- 升龙和波动拳的状态都是 524，要靠有没有离地分开。朝当前脸的方向，升龙第 16 帧离地。面朝左再朝右打，要到第 30 帧才离地。第 16 帧还站在地上就判成波动拳，会把这发慢升龙扣成串招，换边加分也加不上。波动拳会一直站到大约第 55 帧，所以改成第 40 帧还没离地才算波动拳。
- 接进演示时要沿用训练的出招门闩。上一发还没结算完，人即使已经站回地上，新的升龙也不能送出去，否则键会变成轻拳，局面和训练对不上。

各轮的训练配置和最好、最终权重在 `logs/BARREL_*/`。`training_config.json` 里是那一轮的奖励和步数。`TRAINING/ppo_barrel_ryu_best.zip` 是评估最高的一版，`ppo_barrel_ryu_final.zip` 是训完时的权重。中途被停掉的两轮没有最终权重。每 10 万步的中间快照没有放进仓库，十轮合计约 8GB。

常用测试参数如下：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--model-path PATH` | 指定待测试的 PPO 模型权重 | `logs/trained_models/ppo_ryu.zip` |
| `--episodes N` | 逐关评估遍数或连续闯关尝试次数 | `10` |
| `--no-render` | 关闭游戏画面 | 默认显示画面 |
| `--no-load-level-saves` | 从第 1 关连续闯关 | 默认逐关加载存档 |
| `--reset-round` | 逐关测试时，以单次 KO 为评估单位 | 默认三局两胜 |
| `--device auto/cpu/cuda/mps` | 选择模型推理设备 | `auto` |
| `--speed N` | 通过减少显示帧数加快可视化；`--realtime` 时忽略 | `4` |
| `--realtime` | 全屏、原速、有声音 | 关闭 |

### 训练模型

如果想要训练自己的模型，可以运行 `main/train.py`。建议先使用每关一个模拟器的配置，根据机器资源调整训练规模：

```bash
python -m main.train --envs-per-level 1 --batch-size 256 --log-dir runs/ryu
```

直接运行 `python -m main.train` 时，默认每关使用 64 个模拟器，共 768 个；一次更新收集 98,304 个样本，总训练目标为 7 亿步。这一配置需要充足的 CPU 和内存，应根据实际硬件调整。

常用训练参数如下：

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--envs-per-level N` | 指定每关的并行环境数量 | 未指定时每关 64 个 |
| `--total-timesteps N` | 本次训练的总步数；续训时为追加步数 | `700000000` |
| `--n-steps N` | 每个模拟器一次更新前的采样步数 | `128` |
| `--batch-size N` | PPO 更新的批次大小 | `8192` |
| `--n-epochs N` | 每批采样数据的训练轮数 | `3` |
| `--learning-rate N` | 学习率 | `0.0002` |
| `--device auto/cpu/cuda/mps` | 选择模型训练设备 | `mps` |
| `--log-dir PATH` | 模型和训练曲线输出目录 | `logs/ALL_MOTION_SIMULTANEOUS` |

如果想在提供的模型基础上继续训练，可以运行：

```bash
python -m main.train \
  --resume-model logs/trained_models/ppo_ryu.zip \
  --envs-per-level 1 --batch-size 256 \
  --total-timesteps 1000000 --log-dir runs/ryu-resume
```

训练和测试的完整参数可分别通过 `python -m main.train --help` 和 `python -m main.test --help` 查看。也支持 `python main/train.py` 和 `python main/test.py` 直接启动。

### 查看曲线

项目中包含了训练过程的 TensorBoard 曲线，可以查看智能代理在训练中的奖励变化和模型更新情况：

```bash
tensorboard --logdir=logs
```

在浏览器中打开 TensorBoard 服务默认地址 [http://localhost:6006/](http://localhost:6006/)，即可查看训练过程的交互式曲线。

如果想查看自己训练的模型曲线，将日志目录替换为训练时的输出目录即可，例如 `tensorboard --logdir=runs/ryu`。

## 实现说明

游戏画面经过居中裁剪、运动前景提取和缩放，生成 `84×84` 的图像。运动前景提取使用 `t-16、t-8、t` 的灰度差分，保留运动区域并将背景亮度乘以 0.35，分别取 R、G、B 通道，组成 `84×84×3` 的模型输入，帮助 AI 模型感知角色的移动和攻击。

动作空间包含 27 个离散动作，覆盖中立、移动、普通攻击、蹲攻击，以及左右朝向的三种特殊招式。每个动作对应四个模拟器帧的按键序列。

奖励主要根据双方血量变化计算：对对手造成伤害获得正奖励，自身受到伤害产生负奖励，击倒或被击倒还会获得额外奖励或惩罚。伤害奖励为 `0.01 × (3 × 对手掉血 − 自己掉血)`，击倒奖励为 `0.01 × (176 + 自己剩余血量)`，被击倒惩罚为 `0.01 × (176 + 对手剩余血量)`。每次实际执行动作还会扣除时间惩罚，默认值为 0.0005。

## 鸣谢

本项目使用了 [Gymnasium](https://gymnasium.farama.org/)、[Stable-Retro](https://github.com/Farama-Foundation/stable-retro)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 等开源代码库。感谢各位程序工作者对开源社区的贡献。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
