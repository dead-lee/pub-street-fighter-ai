"""认出画面上每一只正在动的桶，一只一个位置。

桶为了转起来，会轮流画几张不同的图：滑槽里的侧面、落地后的圆脸、
顶上只露出一截的半张。墙上码着的那一排不转，不是要打的桶。
这里是一个很小的卷积网络，看每个局部像不像这些桶，一格 4 个像素。
"""

from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

# 输出一格对应画面上的像素边长。桶的中心由这一格换算回来。
BARREL_STRIDE = 4
WEIGHT_PATH = Path(__file__).with_name("barrel_vision.pt")

_net = None
_threshold = None


class BarrelNet(nn.Module):
    """感受野大约一只桶。输出每个格子有多像桶，不限于同一种转法。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image):
        return self.net(image)


def _load():
    global _net, _threshold
    if _net is not None:
        return
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    net = BarrelNet()
    net.load_state_dict(blob["state"])
    net.eval()
    _net = net
    _threshold = float(blob["threshold"])


def barrel_centers(image):
    """这一帧里每只桶的中心。返回 (x, y, 置信度)，可能有多只。"""
    _load()
    if image is None:
        return []
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return []
    frame = np.ascontiguousarray(frame[:, :, :3])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    with torch.no_grad():
        heat = torch.sigmoid(_net(tensor))[0, 0].numpy()
    mask = (heat >= _threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        cx, cy = centroids[i]
        # 格子坐标换成画面像素。格子 0 的中心在像素 stride/2。
        px = float(cx) * BARREL_STRIDE + BARREL_STRIDE / 2
        py = float(cy) * BARREL_STRIDE + BARREL_STRIDE / 2
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        score = float(heat[y0:y0 + h, x0:x0 + w].max())
        found.append((px, py, score))
    found.sort(key=lambda item: -item[2])
    return found[:6]
