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
# 比空墙高出这么多，才算这一帧新出现的元素。画在墙上的圆脸和墙一样亮。
BRIGHT_DELTA = 12.0
# 一格里至少这么多像素比墙亮，这格才算有效元素。桶边的暗线占不满一格。
BRIGHT_COVER = 0.35
WEIGHT_PATH = Path(__file__).with_name("barrel_vision.pt")
# 第二段看原像素上的一块。边长盖住一只圆桶和它旁边的破口。
CROP_SIZE = 72
CROP_PATH = Path(__file__).with_name("barrel_crop.pt")

_net = None
_threshold = None
_crop_net = None
_crop_threshold = None
_crop_ready = False
# 木桶关的空墙。和当前帧比亮度，先分出有效元素。
_background = None
# 上一帧画面。圆桶换一张滚动图时，外框里的像素会变，木坡不会。用这个差找出外框。
_prev_frame = None


def forget_frame():
    """换了一局。上一局的最后一帧不能拿来和这一局相减。"""
    global _prev_frame
    _prev_frame = None


def _brown(frame):
    """桶板的颜色。白衣服、红字、蓝头带不在这里面。"""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    return (
        (red > 75)
        & (green > 40)
        & (blue < 170)
        & (red + 10 > green)
        & (green + 8 > blue)
        & ((red - blue) > 20)
        & (red < 220)
        & (green < 190)
    )


def _round_outline_centers(frame, prev):
    """地上圆桶的外框中心。花纹换帧时，亮着的格子会跟着桶板走，外框不动。"""
    if prev is None or prev.shape != frame.shape:
        return []
    changed = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).sum(axis=2) > 24
    pieces = (changed & _brown(frame)).astype(np.uint8)
    pieces[:, 256:] = 0
    pieces[:16, :] = 0
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(pieces, 8)
    found = []
    for index in range(1, count):
        top = int(stats[index, cv2.CC_STAT_TOP])
        left = int(stats[index, cv2.CC_STAT_LEFT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if top + height / 2 < 150 or not (30 <= width <= 60 and 30 <= height <= 60 and area >= 400):
            continue
        cx, cy = centroids[index]
        found.append((float(cx), float(cy), left, top, left + width, top + height))
    return found


class BarrelNet(nn.Module):
    """每个格子先看大约一只桶。旁边再看一圈，拳头比桶小，这一圈里装不满一只桶。

    旁边这一层从零开始，不改变已经学好的输出，直到用拳头画面把它训开。
    """

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
        self.wide = nn.Conv2d(16, 1, kernel_size=7, padding=3, bias=False)
        nn.init.zeros_(self.wide.weight)

    def forward(self, image):
        features = image
        layers = list(self.net.children())
        for layer in layers[:-1]:
            features = layer(features)
        # 原来那一格的判断，加上周围一圈。零权重时两项里的第二项是 0。
        return layers[-1](features) + self.wide(features)


class BarrelCropNet(nn.Module):
    """原像素上的一块。前两层不缩小，锯齿还在，整块再合成是不是一只桶。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, image):
        return self.head(self.features(image).flatten(1)).squeeze(1)


def crop_at(frame, cx, cy, size=CROP_SIZE):
    """以这个点为中心裁一块。贴边时用黑边补上，不把桶挪到裁块中间以外。"""
    half = size // 2
    x0 = int(round(float(cx))) - half
    y0 = int(round(float(cy))) - half
    canvas = np.zeros((size, size, 3), np.uint8)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(frame.shape[1], x0 + size)
    src_y1 = min(frame.shape[0], y0 + size)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas
    canvas[
        src_y0 - y0:src_y0 - y0 + (src_y1 - src_y0),
        src_x0 - x0:src_x0 - x0 + (src_x1 - src_x0),
    ] = frame[src_y0:src_y1, src_x0:src_x1, :3]
    return canvas


def _crop_model():
    """第二段的权重。文件还没有时，热度点照旧交出去。"""
    global _crop_net, _crop_threshold, _crop_ready
    if _crop_ready:
        return _crop_net, _crop_threshold
    _crop_ready = True
    if not CROP_PATH.is_file():
        return None, None
    blob = torch.load(CROP_PATH, map_location="cpu", weights_only=False)
    net = BarrelCropNet()
    net.load_state_dict(blob["state"])
    net.eval()
    _crop_net = net
    _crop_threshold = float(blob["threshold"])
    return _crop_net, _crop_threshold


def _keep_refined(frame, points):
    """每个热度点裁一块，细网络确认这是整桶才留下。"""
    net, threshold = _crop_model()
    if net is None or not points:
        return points
    crops = np.stack([crop_at(frame, x, y) for x, y, _score in points])
    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div_(255)
    with torch.no_grad():
        prob = torch.sigmoid(net(tensor)).numpy()
    return [(x, y, float(p)) for (x, y, _score), p in zip(points, prob) if float(p) >= threshold]


def _luma(frame):
    """像素有多亮。真桶的桶板比它盖住的那块墙亮。"""
    image = frame.astype(np.float32)
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def _bright_cover(frame, background, heat):
    """这一格里有多少像素比空墙亮。不够的格子不是这一帧的有效元素。"""
    if background is None or background.shape[:2] != frame.shape[:2]:
        return np.ones(heat.shape, np.float32)
    bright = (_luma(frame) - _luma(background) > BRIGHT_DELTA).astype(np.float32)
    height, width = heat.shape
    if bright.shape[0] < height * BARREL_STRIDE or bright.shape[1] < width * BARREL_STRIDE:
        return np.ones(heat.shape, np.float32)
    block = bright[:height * BARREL_STRIDE, :width * BARREL_STRIDE]
    return block.reshape(height, BARREL_STRIDE, width, BARREL_STRIDE).mean(axis=(1, 3))


def _load():
    global _net, _threshold, _background
    if _net is not None:
        return
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    net = BarrelNet()
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    _net = net
    _threshold = float(blob["threshold"])
    background = blob.get("background")
    _background = None if background is None else np.asarray(background, dtype=np.uint8)


def barrel_centers(image):
    """这一帧里每只桶的中心。返回 (x, y, 置信度)，可能有多只。

    滑槽和空中的桶仍用热度格子的中心。地上的圆桶改用外框中心：
    超过门槛的格子只覆盖到这一帧亮着的桶板，质心会跟着纹路上下走。
    """
    global _prev_frame
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
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
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
    outlines = _round_outline_centers(frame, _prev_frame)
    used = set()
    snapped = []
    for px, py, score in found:
        best = None
        best_d = 18.0
        for index, (bx, by, left, top, right, bottom) in enumerate(outlines):
            if index in used:
                continue
            # 热度中心还落在这只圆桶里，才把坐标换成外框。旁边另一只桶不借用这个框。
            if not (left - 6 <= px <= right + 6 and top - 6 <= py <= bottom + 6):
                continue
            distance = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
            if distance < best_d:
                best_d = distance
                best = index
        if best is None:
            snapped.append((px, py, score))
            continue
        used.add(best)
        bx, by = outlines[best][0], outlines[best][1]
        snapped.append((bx, by, score))
    _prev_frame = np.array(frame, copy=True)
    snapped.sort(key=lambda item: -item[2])
    return _keep_refined(frame, snapped[:6])
