"""用三份木桶开局、龙不出招的画面，训练认出每只桶的小网络。

这里只覆盖站立时的标签。升龙把整屏抖起来的那几帧，要另外压过拳头和木梁，
直接跑这个文件会把已经看过的权重换成只有站立画面的那一版。

标签不是手动画的。墙不动，拿中位数当空背景。和背景不一样、又是桶板那种棕色的，
才是正在出现的桶。人的身体按脚的位置涂掉，分数上的亮字也涂掉。
顶上只露出一截的、滑槽里的、落地的圆脸，都留在标签里。
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from main.barrel_vision import BarrelNet, BARREL_STRIDE, WEIGHT_PATH

FRAME_DIR = Path("/tmp/barrel_forms")
# 整屏颜色一起变的是过场，不能拿来当哪一格是桶。
FLASH_FRACTION = 0.18


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


def _ryu_box(agent_x, agent_y, status):
    """人站着或躺着时身体占的那一块。桶贴到人身上时，这一块里的像素不当标签。"""
    if agent_x < 50 or agent_y < 100:
        return None
    rx = 73 + (int(agent_x) - 205) * (10 / 15.0)
    knocked = int(status) == 526
    half = 40 if knocked else 26
    top = 36 if knocked else 72
    return (
        int(rx - half),
        int(agent_y - top),
        int(rx + half),
        int(agent_y + 8),
    )


def label_mask(frame, background, agent_x, agent_y, status):
    """这一帧里桶占的像素。过场闪屏返回空，调用方整帧丢掉。"""
    foreground = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) > 36
    if foreground.mean() > FLASH_FRACTION:
        return None
    mask = (foreground & _brown(frame)).astype(np.uint8)
    # 最上面一行是分数。左边那一块是 1UP，不是桶。
    mask[:16, :] = 0
    mask[:36, :88] = 0
    mask[:, 256:] = 0
    box = _ryu_box(agent_x, agent_y, status)
    if box is not None:
        x0, y0, x1, y1 = box
        mask[max(0, y0):min(mask.shape[0], y1), max(0, x0):min(mask.shape[1], x1)] = 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        # 太碎的是字的残边。比一只桶还大的是人和桶粘在了一起。
        if area < 36 or width < 6 or height < 4 or width > 74 or height > 74:
            continue
        if area / max(width * height, 1) < 0.28:
            continue
        keep[labels == i] = 1
    return keep


def _downsample(mask):
    """4 像素一格，格子里有桶的像素就算这格是桶。"""
    height, width = mask.shape
    trimmed_h = height - height % BARREL_STRIDE
    trimmed_w = width - width % BARREL_STRIDE
    mask = mask[:trimmed_h, :trimmed_w]
    cells_h = trimmed_h // BARREL_STRIDE
    cells_w = trimmed_w // BARREL_STRIDE
    folded = mask.reshape(cells_h, BARREL_STRIDE, cells_w, BARREL_STRIDE)
    return folded.max(axis=(1, 3)).astype(np.float32)


def load_lives():
    frames = []
    metas = []
    backgrounds = []
    for index in range(1, 4):
        live = np.load(FRAME_DIR / f"live{index}.npy")
        meta = np.load(FRAME_DIR / f"live{index}_meta.npy")
        # 开局有几帧还在淡入，中位数从站稳之后算，墙才是空的。
        background = np.median(live[40::4], axis=0).astype(np.uint8)
        frames.append(live)
        metas.append(meta)
        backgrounds.append(background)
    return frames, metas, backgrounds


def build_samples(frames, metas, backgrounds):
    images = []
    targets = []
    which = []
    for life, (live, meta, background) in enumerate(zip(frames, metas, backgrounds)):
        for i in range(40, len(live)):
            mask = label_mask(live[i], background, meta[i, 0], meta[i, 1], meta[i, 2])
            if mask is None:
                continue
            images.append(live[i])
            targets.append(_downsample(mask))
            which.append((life, i))
    return images, targets, which


def train():
    frames, metas, backgrounds = load_lives()
    images, targets, which = build_samples(frames, metas, backgrounds)
    positive = float(np.mean([item.mean() for item in targets]))
    print(f"samples {len(images)} positive cells {positive:.4f}", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = BarrelNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    order = np.arange(len(images))
    batch = 16
    for epoch in range(8):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps = 0
        pos_sum = 0.0
        pos_n = 0
        neg_sum = 0.0
        neg_n = 0
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            batch_x = np.stack([images[i] for i in pick]).astype(np.float32) / 255.0
            batch_y = np.stack([targets[i] for i in pick])
            # 左右翻面。桶从两边滑下来，转法是同一套。
            if rng.random() < 0.5:
                batch_x = batch_x[:, :, ::-1, :].copy()
                batch_y = batch_y[:, :, ::-1].copy()
            tensor_x = torch.from_numpy(batch_x).permute(0, 3, 1, 2).to(device)
            tensor_y = torch.from_numpy(np.ascontiguousarray(batch_y)).unsqueeze(1).to(device)
            logits = net(tensor_x)
            if logits.shape[-2:] != tensor_y.shape[-2:]:
                logits = F.interpolate(logits, size=tensor_y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logits, tensor_y, pos_weight=torch.tensor([6.0], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pos = tensor_y > 0.5
                pos_sum += float(prob[pos].sum())
                pos_n += int(pos.sum())
                neg = ~pos
                neg_sum += float(prob[neg].sum())
                neg_n += int(neg.sum())
        print(
            f"epoch {epoch} loss {total / max(steps, 1):.4f} "
            f"pos {pos_sum / max(pos_n, 1):.3f} neg {neg_sum / max(neg_n, 1):.3f}",
            flush=True,
        )
    net.eval()
    # 空墙上的最高分，和真桶上的最低分，门槛放在两者中间。
    wall_peak = 0.0
    barrel_floor = 1.0
    with torch.no_grad():
        for life, background in enumerate(backgrounds):
            tensor = torch.from_numpy(background).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
            wall_peak = max(wall_peak, float(torch.sigmoid(net(tensor)).max()))
        for i in range(0, len(images), 25):
            if targets[i].max() < 0.5:
                continue
            tensor = torch.from_numpy(images[i]).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
            prob = torch.sigmoid(net(tensor))[0, 0].cpu().numpy()
            hot = targets[i] > 0.5
            if hot.any():
                barrel_floor = min(barrel_floor, float(prob[hot].max()))
    threshold = float(np.clip((wall_peak + barrel_floor) / 2, 0.35, 0.75))
    print(f"wall {wall_peak:.3f} barrel {barrel_floor:.3f} threshold {threshold:.3f}", flush=True)
    blob = {"state": {key: value.detach().cpu() for key, value in net.state_dict().items()}, "threshold": threshold}
    # 不写进正在用的权重。那一份已经把升龙帧里的拳头和木梁压掉了。
    staging = WEIGHT_PATH.with_name("barrel_vision_standing.pt")
    torch.save(blob, staging)
    print(f"wrote {staging}", flush=True)
    return net.cpu(), threshold, images, targets, which, backgrounds


def _solidity(component):
    """越接近 1 越像实心的整桶。碎桶板是凹进去的半片。"""
    contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 1.0
    return float(cv2.contourArea(contour) / hull_area)


def _fragment_mask(frame, background):
    """竖着的半片桶板。圆脸整桶又圆又实，不在这里面。"""
    foreground = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) > 36
    if foreground.mean() > FLASH_FRACTION:
        return None
    pieces = (foreground & _brown(frame)).astype(np.uint8)
    pieces[:, 256:] = 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(pieces, 8)
    found = np.zeros_like(pieces)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        # 半片贴在地上，又窄又高。圆脸大约四十像素见方，实心度在 0.9 以上。
        if y < 140 or width > 34 or height < 40 or not (200 <= area <= 1000):
            continue
        if _solidity((labels == index).astype(np.uint8)) >= 0.82:
            continue
        found[labels == index] = 1
    return found


def teach_fragments():
    """碎桶板原来和整桶标在一起，模型会把它当成还能打的桶。

    从正在用的权重接着学。半片压成不是桶，滑槽里的、圆脸、顶上露出的一截都留着。
    学完如果整桶还在、半片掉下去了，才写回 barrel_vision.pt。
    """
    frames, metas, backgrounds = load_lives()
    images = []
    targets = []
    weights = []
    held_out = []
    for life, (live, meta, background) in enumerate(zip(frames, metas, backgrounds)):
        for index in range(40, len(live), 2):
            pieces = _fragment_mask(live[index], background)
            if pieces is None:
                continue
            positive = label_mask(live[index], background, meta[index, 0], meta[index, 1], meta[index, 2])
            if positive is None:
                continue
            positive = positive.copy()
            positive[pieces > 0] = 0
            target = _downsample(positive)
            hard = _downsample(pieces)
            target[hard > 0] = 0
            weight = np.ones_like(target)
            # 半片格子很少。权重太小的话，整屏的空地会把这个反例淹没。
            weight[hard > 0] = 20.0
            sample = (live[index], target, weight)
            if life == 2:
                held_out.append(sample)
            else:
                images.append(sample[0])
                targets.append(sample[1])
                weights.append(sample[2])
    # 波动拳那一帧：半片是反例，人的身体也是反例。身体按画面中心涂，不能用压缩后的横坐标，否则右臂会剩在外面被标成桶。
    play = np.load("/tmp/hitheight/play_frags.npy")
    for frame in play:
        pieces = _fragment_mask(frame, backgrounds[0])
        positive = label_mask(frame, backgrounds[0], 245, 192, 512)
        if pieces is None or positive is None:
            continue
        positive = positive.copy()
        positive[pieces > 0] = 0
        body = np.zeros(positive.shape, np.uint8)
        body[110:200, 70:155] = 1
        positive[body > 0] = 0
        target = _downsample(positive)
        hard = _downsample(np.maximum(pieces, body))
        target[hard > 0] = 0
        weight = np.ones_like(target)
        weight[_downsample(pieces) > 0] = 20.0
        weight[_downsample(body) > 0] = np.maximum(weight[_downsample(body) > 0], 8.0)
        if int(pieces.sum()) < 100:
            continue
        for _repeat in range(4):
            images.append(frame)
            targets.append(target)
            weights.append(weight)
    # 只把真有半片的帧多放几遍。滑槽桶那一格权重是 1，不能拿来当半片。
    extra_i = [index for index, weight in enumerate(weights) if float((weight > 1).sum()) > 0]
    base_n = len(extra_i)
    for _repeat in range(8):
        for index in extra_i[:base_n]:
            images.append(images[index])
            targets.append(targets[index])
            weights.append(weights[index])
    fragment_cells = int(sum(float((weight > 1).sum()) for weight in weights))
    print(f"train {len(images)} held {len(held_out)} fragment cells {fragment_cells} extra {len(extra_i)}", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    backup = Path("/tmp/barrel_vision_before_frag.pt")
    torch.save(blob, backup)
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"])
    threshold = float(blob["threshold"])

    def fragment_peak(samples):
        peaks = []
        rounds = []
        with torch.no_grad():
            for image, target, weight in samples[::8]:
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
                prob = torch.sigmoid(net(tensor))[0, 0]
                if prob.shape != weight.shape:
                    prob = F.interpolate(prob[None, None], size=weight.shape, mode="bilinear", align_corners=False)[0, 0]
                hard = torch.from_numpy(weight).to(device) > 1
                whole = torch.from_numpy(target).to(device) > 0.5
                if hard.any():
                    peaks.append(float(prob[hard].max()))
                if whole.any():
                    rounds.append(float(prob[whole].max()))
        return (
            max(peaks) if peaks else 0.0,
            min(rounds) if rounds else 0.0,
        )

    print("before", fragment_peak(held_out), flush=True)
    play_frame = play[1]
    round_frame = frames[0][250]
    # 这一帧滑槽里有整桶，地上还没有圆脸，用来确认半片被压掉时滑槽桶还在。
    chute_frame = frames[0][200]

    def region_peak(image, x0, y0, x1, y1):
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
        with torch.no_grad():
            heat = torch.sigmoid(net(tensor))[0, 0]
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    print(
        "start",
        "left", round(region_peak(play_frame, 30, 140, 80, 200), 3),
        "right", round(region_peak(play_frame, 180, 140, 240, 200), 3),
        "chute", round(region_peak(chute_frame, 90, 10, 180, 80), 3),
        "round", round(region_peak(round_frame, 120, 140, 200, 200), 3),
        "body", round(region_peak(round_frame, 50, 100, 120, 170), 3),
        flush=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    order = np.arange(len(images))
    batch = 16
    pos_weight = torch.tensor([6.0], device=device)
    for epoch in range(12):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            batch_x = np.stack([images[i] for i in pick]).astype(np.float32) / 255.0
            batch_y = np.stack([targets[i] for i in pick])
            batch_w = np.stack([weights[i] for i in pick])
            if rng.random() < 0.5:
                batch_x = batch_x[:, :, ::-1, :].copy()
                batch_y = batch_y[:, :, ::-1].copy()
                batch_w = batch_w[:, :, ::-1].copy()
            tensor_x = torch.from_numpy(np.ascontiguousarray(batch_x)).permute(0, 3, 1, 2).to(device)
            tensor_y = torch.from_numpy(np.ascontiguousarray(batch_y)).unsqueeze(1).to(device)
            tensor_w = torch.from_numpy(np.ascontiguousarray(batch_w)).unsqueeze(1).to(device)
            logits = net(tensor_x)
            if logits.shape[-2:] != tensor_y.shape[-2:]:
                logits = F.interpolate(logits, size=tensor_y.shape[-2:], mode="bilinear", align_corners=False)
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, pos_weight=pos_weight, reduction="none")
            loss = (loss_map * tensor_w).sum() / tensor_w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        chute = region_peak(chute_frame, 90, 10, 180, 80)
        whole = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        print(
            f"epoch {epoch} loss {total / max(steps, 1):.4f} "
            f"left {left:.3f} right {right:.3f} chute {chute:.3f} round {whole:.3f} "
            f"body {body:.3f} arm {arm:.3f}",
            flush=True,
        )
        # 半片掉下去，滑槽桶和圆脸还在，人身上不能冒出桶。
        if left < 0.55 and right < 0.55 and chute >= 0.85 and whole >= 0.85 and body < 0.45 and arm < 0.45:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
    print("reject, keep old weights", flush=True)


if __name__ == "__main__":
    train()
