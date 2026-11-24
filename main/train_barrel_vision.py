"""用三份木桶开局、龙不出招的画面，训练认出每只桶的小网络。

这里只覆盖站立时的标签。升龙把整屏抖起来的那几帧，要另外压过拳头和木梁，
直接跑这个文件会把已经看过的权重换成只有站立画面的那一版。

标签不是手动画的。墙不动，拿中位数当空背景。和背景不一样、又是桶板那种棕色的，
才是正在出现的桶。人的身体按脚的位置涂掉，分数上的亮字也涂掉。
顶上只露出一截的、滑槽里的、落地的圆脸，都留在标签里。
"""

from pathlib import Path
import json

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


def _round_ground_mask(frame, background):
    """地上滚动的整桶。外框大约四十像素见方、又实。花纹换帧时仍是同一只。"""
    foreground = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) > 36
    pieces = (foreground & _brown(frame)).astype(np.uint8)
    pieces[:, 256:] = 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(pieces, 8)
    found = np.zeros_like(pieces)
    for index in range(1, count):
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        # 用外框中心，不用上沿。圆桶高四十多像素，上沿会高过 150。
        if y + height / 2 < 158 or not (36 <= width <= 52 and 36 <= height <= 52 and area >= 700):
            continue
        component = (labels == index).astype(np.uint8)
        if _solidity(component) < 0.85:
            continue
        found[component > 0] = 1
    return found


LOCKED_DIR = Path(__file__).resolve().parents[1] / "data" / "barrel_locked"


def _locked_spec():
    """确认过的样本。正面、墙上的桶、火球、左右半片都在这里，训练时照单使用。"""
    return json.loads((LOCKED_DIR / "samples.json").read_text(encoding="utf-8"))


def _box_mask(shape, boxes):
    mask = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[int(y0):int(y1), int(x0):int(x1)] = 1
    return mask


def _locked_frame(name):
    return cv2.cvtColor(cv2.imread(str(LOCKED_DIR / name)), cv2.COLOR_BGR2RGB)


def _stage_background(frame, backgrounds):
    """挑和这张画面墙面最像的空舞台。半片要贴回原来的位置，不能贴到另一关的墙上。"""
    best = None
    best_score = None
    for background in backgrounds:
        if background.shape[:2] != frame.shape[:2]:
            continue
        diff = np.abs(frame[:, :, :3].astype(np.int16) - background[:, :, :3].astype(np.int16)).sum(axis=2)
        score = float(diff[20:110, 40:220].mean())
        if best_score is None or score < best_score:
            best_score = score
            best = background
    return best


def _piece_alone(frame, background, box):
    """一只半片贴在空舞台原来的位置。另一只和人都不在这张图里。"""
    alone = np.array(background[:, :, :3], copy=True)
    x0, y0, x1, y1 = (int(value) for value in box)
    alone[y0:y1, x0:x1] = frame[y0:y1, x0:x1, :3]
    return alone


def _jagged_box(frame, box):
    """破口上的方框。左右两条边里更碎的那条是锯齿，方框只盖住齿尖到齿根。"""
    x0, y0, x1, y1 = (int(value) for value in box)
    wood = _brown(frame[y0:y1, x0:x1])
    left, right = [], []
    for row in range(wood.shape[0]):
        cols = np.flatnonzero(wood[row])
        if len(cols) < 2:
            continue
        left.append(int(cols[0]))
        right.append(int(cols[-1]))
    if len(left) < 6:
        return None
    def rough(edge):
        return float(np.mean(np.abs(np.diff(np.asarray(edge, np.float32)))))
    use_right = rough(right) >= rough(left)
    edge = np.asarray(right if use_right else left)
    inner = int(edge.min())
    outer = int(edge.max()) + 1
    mid = wood.shape[1] // 2
    if use_right:
        inner = max(inner - 2, mid)
        box_x0 = x0 + inner
        box_x1 = x0 + min(wood.shape[1], outer)
    else:
        outer = min(outer + 2, mid)
        box_x0 = x0 + max(0, inner)
        box_x1 = x0 + outer
    if box_x1 - box_x0 < 4:
        return None
    return [box_x0, y0, box_x1, y1]


def _jagged_boxes(frame, mask):
    """每一只半片各给破口一个方框。圆边留在框外。"""
    if mask is None or int(np.asarray(mask).sum()) < 1:
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    boxes = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        box = _jagged_box(frame, [x, y, x + width, y + height])
        if box is not None:
            boxes.append(box)
    return boxes


def _ground_fragments(frame, background, whole=None):
    """贴地的半片，左右都算。整只圆桶不在这里。

    只框半片自己那一块。不往旁边的整桶上扩。
    """
    foreground = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) > 36
    pieces = (foreground & _brown(frame)).astype(np.uint8)
    pieces[:, 256:] = 0
    pieces[:16, :] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(pieces, 8)
    found = np.zeros_like(pieces)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if float(centroids[index][1]) < 170:
            continue
        if not (10 <= width <= 34 and 16 <= height <= 52 and 120 <= area <= 950):
            continue
        component = (labels == index).astype(np.uint8)
        if _solidity(component) >= 0.82:
            continue
        if whole is not None and float(whole[y:y + height, x:x + width].mean()) > 0.3:
            continue
        found[component > 0] = 1
    return found


def _fragment_mask(frame, background):
    """地上的破桶。圆脸整桶又圆又实，不在这里面。

    破裂的缺口往左右伸出去。半片本体标成反例之后，再把左右各 12 像素里的桶板色也标成反例，
    裂口才算进负分。圆脸不跟着被涂掉。
    """
    foreground = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) > 36
    pieces = (foreground & _brown(frame)).astype(np.uint8)
    pieces[:, 256:] = 0
    found = np.zeros_like(pieces)
    rounds = np.zeros_like(pieces)
    raw_count, raw_labels, raw_stats, _raw_centroids = cv2.connectedComponentsWithStats(pieces, 8)
    for index in range(1, raw_count):
        width = int(raw_stats[index, cv2.CC_STAT_WIDTH])
        height = int(raw_stats[index, cv2.CC_STAT_HEIGHT])
        if width < 36 or height < 28:
            continue
        if _solidity((raw_labels == index).astype(np.uint8)) < 0.85:
            continue
        rounds[raw_labels == index] = 1
    # 破桶常常和斜坡粘成一块。先留下竖着的窄条，再往左右各看 12 像素。
    # 裂口在这 12 像素里，实心度会掉下去；圆脸扩出去还是实的，不当反例。
    opened = cv2.morphologyEx(pieces, cv2.MORPH_OPEN, np.ones((5, 3), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, 8)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if y + height < 148 or width > 34 or height < 40 or not (200 <= area <= 1000):
            continue
        x0 = max(0, x - 12)
        x1 = min(pieces.shape[1], x + width + 12)
        y1 = min(pieces.shape[0], y + height)
        wide = pieces[y:y1, x0:x1]
        if int(wide.sum()) < 200 or _solidity(wide) >= 0.82:
            continue
        band = np.zeros_like(pieces)
        band[y:y1, x0:x1] = wide
        found[(band > 0) & (rounds == 0)] = 1
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
    # 当前对局里被认成整桶的半片。身体按画面中心涂掉，半片格子当反例。
    edge_probe = None
    edge_path = Path("/tmp/hitheight/edge_pieces.npz")
    if edge_path.exists():
        pack = np.load(edge_path)
        best_pixels = 0
        for frame, ax, ay, status, life in zip(pack["frames"], pack["ax"], pack["ay"], pack["status"], pack["life"]):
            background = backgrounds[int(life)]
            pieces = _fragment_mask(frame, background)
            positive = label_mask(frame, background, int(ax), int(ay), int(status))
            if pieces is None or positive is None or int(pieces.sum()) < 100:
                continue
            positive = positive.copy()
            positive[pieces > 0] = 0
            body = np.zeros(positive.shape, np.uint8)
            if 90 <= int(ax) <= 420:
                body_x = int(73 + (int(ax) - 205))
                x0 = max(0, body_x - 42)
                x1 = min(body.shape[1], body_x + 38)
                body[110:200, x0:x1] = 1
                positive[body > 0] = 0
            target = _downsample(positive)
            hard = _downsample(np.maximum(pieces, body))
            target[hard > 0] = 0
            weight = np.ones_like(target)
            weight[_downsample(pieces) > 0] = 20.0
            if body.any():
                weight[_downsample(body) > 0] = np.maximum(weight[_downsample(body) > 0], 8.0)
            for _repeat in range(6):
                images.append(frame)
                targets.append(target)
                weights.append(weight)
            right_pixels = int(pieces[155:216, 180:240].sum())
            if int(life) == 1 and right_pixels > best_pixels:
                best_pixels = right_pixels
                edge_probe = frame
    # 11.03 秒左边那片破桶。裂口在左右，和上面的半片一起当反例。
    break_probe = None
    break_path = Path("/tmp/hitheight/break_1103.npz")
    if break_path.exists():
        pack = np.load(break_path)
        best_pixels = 0
        for frame, ax, ay, status in zip(pack["frames"], pack["ax"], pack["ay"], pack["status"]):
            pieces = _fragment_mask(frame, backgrounds[0])
            positive = label_mask(frame, backgrounds[0], int(ax), int(ay), int(status))
            if pieces is None or int(pieces.sum()) < 100:
                continue
            # 出招把整屏差值拉高时，整桶标签会空着。破桶那一块仍然要当反例。
            if positive is None:
                positive = np.zeros(pieces.shape, np.uint8)
            positive = positive.copy()
            positive[pieces > 0] = 0
            body = np.zeros(positive.shape, np.uint8)
            if 90 <= int(ax) <= 420:
                body_x = int(73 + (int(ax) - 205))
                x0 = max(0, body_x - 42)
                x1 = min(body.shape[1], body_x + 38)
                body[110:200, x0:x1] = 1
                positive[body > 0] = 0
            target = _downsample(positive)
            hard = _downsample(np.maximum(pieces, body))
            target[hard > 0] = 0
            weight = np.ones_like(target)
            weight[_downsample(pieces) > 0] = 20.0
            if body.any():
                weight[_downsample(body) > 0] = np.maximum(weight[_downsample(body) > 0], 8.0)
            for _repeat in range(8):
                images.append(frame)
                targets.append(target)
                weights.append(weight)
            left_pixels = int(pieces[120:190, 0:70].sum())
            if left_pixels > best_pixels:
                best_pixels = left_pixels
                break_probe = frame
    # 滚动的圆桶。整只都标成桶，花纹换帧时中心才不跟着某一块桶板跑。
    roll_path = Path("/tmp/hitheight/roll_centers.npz")
    if roll_path.exists():
        pack = np.load(roll_path)
        for frame, ax, ay, status, life in zip(
            pack["frames"], pack["ax"], pack["ay"], pack["status"], pack["life"]
        ):
            if int(life) == 2:
                continue
            background = backgrounds[int(life)]
            rounds = _round_ground_mask(frame, background)
            if int(rounds.sum()) < 400:
                continue
            positive = label_mask(frame, background, int(ax), int(ay), int(status))
            if positive is None:
                positive = np.zeros(rounds.shape, np.uint8)
            positive = positive.copy()
            positive[rounds > 0] = 1
            body = np.zeros(positive.shape, np.uint8)
            if 90 <= int(ax) <= 420:
                body_x = int(73 + (int(ax) - 205))
                x0 = max(0, body_x - 42)
                x1 = min(body.shape[1], body_x + 38)
                body[110:200, x0:x1] = 1
                positive[body > 0] = 0
                rounds = rounds.copy()
                rounds[body > 0] = 0
            if int(rounds.sum()) < 400:
                continue
            target = _downsample(positive)
            weight = np.ones_like(target)
            # 整只圆桶都要亮过门槛。只亮花纹的话，二值化之后的中心会跟着纹路走。
            weight[_downsample(rounds) > 0] = 8.0
            for _repeat in range(4):
                images.append(frame)
                targets.append(target)
                weights.append(weight)
    # 这几只是圆脸和滑槽桶。半片往下压的时候，它们的格子要按住，不能跟着掉。
    keepers = []
    for life, index, x0, y0, x1, y1 in (
        (0, 250, 120, 140, 200, 200),
        (0, 200, 90, 20, 180, 80),
        (2, 480, 210, 145, 256, 200),
        (0, 1072, 140, 145, 190, 200),
        (2, 1048, 60, 145, 120, 200),
    ):
        frame = frames[life][index]
        ax, ay, status = [int(v) for v in metas[life][index, :3]]
        positive = label_mask(frame, backgrounds[life], ax, ay, status)
        if positive is None:
            continue
        target = _downsample(positive)
        weight = np.ones_like(target)
        weight[target > 0.5] = 12.0
        for _repeat in range(4):
            images.append(frame)
            targets.append(target)
            weights.append(weight)
        keepers.append((frame, x0, y0, x1, y1))
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
    net.load_state_dict(blob["state"], strict=False)
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

    def edge_points_peak():
        # 每一块半片自己的格子都要掉下去，不能只看右下角那一帧。
        pack = np.load(edge_path)
        worst = 0.0
        with torch.no_grad():
            for frame, life in zip(pack["frames"], pack["life"]):
                pieces = _fragment_mask(frame, backgrounds[int(life)])
                if pieces is None:
                    continue
                hard = _downsample(pieces) > 0
                if not hard.any():
                    continue
                tensor = torch.from_numpy(np.ascontiguousarray(frame[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
                prob = torch.sigmoid(net(tensor))[0, 0]
                if prob.shape != hard.shape:
                    prob = F.interpolate(prob[None, None], size=hard.shape, mode="bilinear", align_corners=False)[0, 0]
                worst = max(worst, float(prob[torch.from_numpy(hard).to(device)].max()))
        return worst

    def break_points_peak():
        # 斜坡上和地上的破桶，连同左右裂口，每一格都要掉到门槛下。
        if not break_path.exists():
            return 1.0
        pack = np.load(break_path)
        worst = 0.0
        saw = False
        with torch.no_grad():
            for frame in pack["frames"]:
                pieces = _fragment_mask(frame, backgrounds[0])
                if pieces is None:
                    continue
                hard = _downsample(pieces) > 0
                if not hard.any():
                    continue
                saw = True
                tensor = torch.from_numpy(np.ascontiguousarray(frame[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
                prob = torch.sigmoid(net(tensor))[0, 0]
                if prob.shape != hard.shape:
                    prob = F.interpolate(prob[None, None], size=hard.shape, mode="bilinear", align_corners=False)[0, 0]
                worst = max(worst, float(prob[torch.from_numpy(hard).to(device)].max()))
        return worst if saw else 1.0

    def roll_shift():
        # 外框不动时，认出来的中心也不能跟着花纹上下跑。第三份存档没拿来学。
        path = Path("/tmp/hitheight/roll_centers.npz")
        if not path.exists():
            return 99.0
        pack = np.load(path)
        worst = 0.0
        paired = 0
        missed = 0
        with torch.no_grad():
            for frame, life in zip(pack["frames"], pack["life"]):
                if int(life) != 2:
                    continue
                mask = _round_ground_mask(frame, backgrounds[2])
                if int(mask.sum()) < 400:
                    continue
                count, _labels, _stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
                tensor = torch.from_numpy(np.ascontiguousarray(frame[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
                heat = torch.sigmoid(net(tensor))[0, 0].detach().cpu().numpy()
                hot = (heat >= 0.7).astype(np.uint8)
                n_count, _nlabels, _nstats, ncent = cv2.connectedComponentsWithStats(hot, 8)
                nets = []
                for index in range(1, n_count):
                    cx, cy = ncent[index]
                    nets.append((float(cx) * BARREL_STRIDE + BARREL_STRIDE / 2, float(cy) * BARREL_STRIDE + BARREL_STRIDE / 2))
                for index in range(1, count):
                    mx, my = centroids[index]
                    if not nets:
                        missed += 1
                        continue
                    px, py = min(nets, key=lambda item: abs(item[0] - mx) + abs(item[1] - my))
                    # 最近的点已经是另一只桶，这帧不拿来比中心。
                    if abs(px - mx) > 14 or abs(py - my) > 16:
                        missed += 1
                        continue
                    paired += 1
                    worst = max(worst, abs(py - my))
        if paired == 0 or missed > paired:
            return 99.0
        return worst

    print(
        "start",
        "left", round(region_peak(play_frame, 30, 140, 80, 200), 3),
        "right", round(region_peak(play_frame, 180, 140, 240, 200), 3),
        "chute", round(region_peak(chute_frame, 90, 10, 180, 80), 3),
        "round", round(region_peak(round_frame, 120, 140, 200, 200), 3),
        "body", round(region_peak(round_frame, 50, 100, 120, 170), 3),
        "edge", round(region_peak(edge_probe, 180, 155, 240, 216), 3) if edge_probe is not None else None,
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
        edge = region_peak(edge_probe, 180, 155, 240, 216) if edge_probe is not None else 1.0
        pieces = edge_points_peak()
        kept = min(region_peak(image, x0, y0, x1, y1) for image, x0, y0, x1, y1 in keepers) if keepers else 0.0
        # 左边破桶连同左右裂口。这一块还在门槛上就不能换权重。
        broken = break_points_peak()
        roll = roll_shift()
        print(
            f"epoch {epoch} loss {total / max(steps, 1):.4f} "
            f"left {left:.3f} right {right:.3f} chute {chute:.3f} round {whole:.3f} "
            f"body {body:.3f} arm {arm:.3f} edge {edge:.3f} pieces {pieces:.3f} kept {kept:.3f} broken {broken:.3f} roll {roll:.2f}",
            flush=True,
        )
        # 半片掉下去，滑槽桶和圆脸还在，人身上不能冒出桶。滚动的圆桶中心也不能跟着花纹跑。
        if left < 0.55 and right < 0.55 and chute >= 0.85 and whole >= 0.85 and body < 0.45 and arm < 0.45 and edge < 0.55 and pieces < 0.55 and kept >= 0.85 and broken < 0.55 and roll < 3.0:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
    print("reject, keep old weights", flush=True)


def _heat_of(net, frame, device):
    tensor = torch.from_numpy(np.ascontiguousarray(frame[:, :, :3])).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
    with torch.no_grad():
        return torch.sigmoid(net(tensor))[0, 0].detach().cpu().numpy()


def _block_mean(mask, heat):
    """每一格里有多少像素满足条件。用平均，不用“有一个像素就算”，避免桶边被当成空墙。"""
    height, width = heat.shape
    block = mask[:height * BARREL_STRIDE, :width * BARREL_STRIDE]
    return block.reshape(height, BARREL_STRIDE, width, BARREL_STRIDE).mean(axis=(1, 3))


def _body_slice(agent_x):
    """人的身体按身体中心涂掉。压缩后的横坐标偏左，右拳会剩在外面。"""
    body_x = int(73 + (int(agent_x) - 205))
    return body_x - 42, 110, body_x + 38, 200


def _false_wall_cells(heat, positive, frame, background, agent_x):
    """热度高、又不是标签里的桶。这一格大部分像素和空墙一样，才当背景误报。"""
    if positive is None or heat.shape[0] == 0:
        return np.zeros_like(heat, np.float32)
    diff = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2)
    # 整格都和空墙一样才算。桶边只沾到一两个像素的，不能算成画在墙上。
    mean_diff = _block_mean(diff.astype(np.float32), heat)
    labeled = _block_mean(positive.astype(np.float32), heat)
    near = cv2.dilate(positive, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    beside = _block_mean(near.astype(np.float32), heat)
    found = (heat >= 0.7) & (mean_diff < 20) & (labeled < 0.25) & (beside < 0.05)
    # 滑槽口的桶会停住。贴地的圆桶中心在 168 上下，也不在这截墙上。
    found[:90 // BARREL_STRIDE, :] = False
    found[145 // BARREL_STRIDE:, :] = False
    x0, y0, x1, y1 = _body_slice(agent_x)
    found[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] = False
    return found.astype(np.float32)


def _missed_barrel_cells(heat, positive, agent_x):
    """标签里的整桶，这一帧热度不到 0.7。拳头那么小的一块不算。"""
    found = np.zeros_like(heat, np.float32)
    if positive is None:
        return found
    x0, y0, x1, y1 = _body_slice(agent_x)
    body = np.zeros(positive.shape, np.uint8)
    body[y0:y1, x0:x1] = 1
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(positive, 8)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < 200:
            continue
        ys, xs = np.where(labels == index)
        if float(body[ys, xs].mean()) > 0.3:
            continue
        cy, cx = ys // BARREL_STRIDE, xs // BARREL_STRIDE
        if cy.max() >= heat.shape[0] or cx.max() >= heat.shape[1]:
            continue
        if float(heat[cy, cx].max()) >= 0.7:
            continue
        component = np.zeros(positive.shape, np.uint8)
        component[ys, xs] = 1
        covered = _block_mean(component.astype(np.float32), heat)
        found[(covered > 0.25) & (heat >= 0.2)] = 1.0
    return found


def teach_gaps():
    """接着当前权重，只学两类已经出错的格子。

    背景被看成桶：空墙像素上的热度压到门槛下。
    漏看：地上整桶热度不够 0.9 的格子抬上去，避免一拍没报出来。
    已经认对的格子权重是 0，不再整屏重学。半片、滑槽、圆脸、人身上对不上就不写权重。
    """
    frames, metas, backgrounds = load_lives()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    torch.save(blob, Path("/tmp/barrel_vision_before_gap.pt"))
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    threshold = float(blob["threshold"])

    play_frame = np.load("/tmp/hitheight/play_frags.npy")[1]
    round_frame = frames[0][250]
    chute_frame = frames[0][200]
    edge_pack = np.load("/tmp/hitheight/edge_pieces.npz")
    edge_probe = edge_pack["frames"][0]
    keepers = (
        (frames[0][250], 120, 140, 200, 200),
        (frames[0][200], 90, 20, 180, 80),
        (frames[2][480], 210, 145, 256, 200),
        (frames[0][1072], 140, 145, 190, 200),
        (frames[2][1048], 60, 145, 120, 200),
    )

    def region_peak(image, x0, y0, x1, y1):
        heat = _heat_of(net, image, device)
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    def piece_peak(pack_path, background):
        pack = np.load(pack_path)
        worst = 0.0
        saw = False
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)] if pack_path.endswith("edge_pieces.npz") else background)
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = _heat_of(net, frame, device)
            if hard.shape != heat.shape or not hard.any():
                continue
            saw = True
            worst = max(worst, float(heat[hard].max()))
        return worst if saw else 1.0

    # 第三关人站在画死的圆脸前面、空墙被看成桶的格子。这些帧不拿来学。
    holdout_points = []
    live, meta, background = frames[2], metas[2], backgrounds[2]
    for index in range(40, len(live), 4):
        positive = label_mask(live[index], background, int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
        if positive is None:
            continue
        heat = _heat_of(net, live[index], device)
        false_cells = _false_wall_cells(heat, positive, live[index], background, int(meta[index, 0]))
        ys, xs = np.where(false_cells > 0)
        for y, x in zip(ys, xs):
            holdout_points.append((index, int(y), int(x), float(heat[y, x])))
    if not holdout_points:
        raise RuntimeError("第三关没有留下墙上的误报可以验收")
    np.save("/tmp/hitheight/wall_holdout.npy", live[max(holdout_points, key=lambda item: item[3])[0]])

    images, targets, weights = [], [], []
    train_wall = []
    # 第一、二关的空闲画面用来学。第三关的漏看和背景误报留着验收。
    for life in (0, 1):
        live, meta, background = frames[life], metas[life], backgrounds[life]
        for index in range(40, len(live), 4):
            # 这两帧的圆桶留着验收漏看，不拿来学。
            if life == 0 and index in (800, 808):
                continue
            positive = label_mask(live[index], background, int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
            if positive is None:
                continue
            heat = _heat_of(net, live[index], device)
            false_cells = _false_wall_cells(heat, positive, live[index], background, int(meta[index, 0]))
            missed = _missed_barrel_cells(heat, positive, int(meta[index, 0]))
            if float(false_cells.sum() + missed.sum()) < 1 and index % 16 != 0:
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            # 已经认对的桶心按住。不然把墙上的圆脸压下去时，真桶会一起掉。
            labeled = _block_mean(positive.astype(np.float32), heat) > 0.5
            keep = labeled & (heat >= 0.85)
            target[keep] = 1.0
            weight[keep] = 3.0
            target[missed > 0] = 1.0
            weight[missed > 0] = 20.0
            weight[false_cells > 0] = 80.0
            copies = 3 if float(false_cells.sum() + missed.sum()) >= 1 else 1
            for _copy in range(copies):
                images.append(live[index])
                targets.append(target)
                weights.append(weight)
            ys, xs = np.where(false_cells > 0)
            for y, x in zip(ys, xs):
                train_wall.append((live[index], int(y), int(x)))
    # 滑槽和圆脸只按住已经在门槛上的格子，不整屏重学。
    for frame, x0, y0, x1, y1 in keepers:
        heat = _heat_of(net, frame, device)
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        y_cells, x_cells = np.where(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] >= 0.85)
        target[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 1.0
        weight[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 8.0
        if float(weight.sum()) < 1:
            continue
        images.append(frame)
        targets.append(target)
        weights.append(weight)
    # 人身上和半片本来就在门槛下。放进损失里，避免把棕色都抬成桶。
    for frame, x0, y0, x1, y1 in ((round_frame, 50, 100, 120, 170), (play_frame, 70, 110, 155, 175)):
        heat = _heat_of(net, frame, device)
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        weight[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] = 16.0
        images.append(frame)
        targets.append(target)
        weights.append(weight)
    for pack_path in ("/tmp/hitheight/edge_pieces.npz", "/tmp/hitheight/break_1103.npz"):
        pack = np.load(pack_path)
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        added = 0
        for frame, life in zip(pack["frames"], lives):
            if added >= 4:
                break
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = _heat_of(net, frame, device)
            if hard.shape != heat.shape or not hard.any():
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            weight[hard] = 24.0
            images.append(frame)
            targets.append(target)
            weights.append(weight)
            added += 1
    cold_tests = []
    live, meta, background = frames[0], metas[0], backgrounds[0]
    for index in (800, 808):
        positive = label_mask(live[index], background, int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
        if positive is None:
            continue
        heat = _heat_of(net, live[index], device)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(positive, 8)
        for comp in range(1, count):
            if int(stats[comp, cv2.CC_STAT_AREA]) < 200:
                continue
            ys, xs = np.where(labels == comp)
            if float(heat[ys // BARREL_STRIDE, xs // BARREL_STRIDE].max()) >= 0.7:
                continue
            cold_tests.append((0, index, float(centroids[comp][0]), float(centroids[comp][1])))
    if not cold_tests:
        raise RuntimeError("没有留下漏看的圆桶可以验收")
    print(
        f"gap samples {len(images)} wall {len(holdout_points)} cold {len(cold_tests)}",
        flush=True,
    )

    def report(tag):
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        chute = region_peak(chute_frame, 90, 10, 180, 80)
        whole = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        edge = region_peak(edge_probe, 180, 155, 240, 216)
        pieces = piece_peak("/tmp/hitheight/edge_pieces.npz", backgrounds[0])
        broken = piece_peak("/tmp/hitheight/break_1103.npz", backgrounds[0])
        kept = min(region_peak(image, x0, y0, x1, y1) for image, x0, y0, x1, y1 in keepers)
        wall = 0.0
        for index, y, x, _before in holdout_points:
            wall = max(wall, float(_heat_of(net, frames[2][index], device)[y, x]))
        trained_wall = 0.0
        for picture, y, x in train_wall:
            trained_wall = max(trained_wall, float(_heat_of(net, picture, device)[y, x]))
        wall = wall  # 验收用第三关。训练格子的热度一起打出来，看是没学到还是没泛化。
        print(f"  train_wall {trained_wall:.3f}", flush=True)
        # 第三关里、训练前热度不到门槛的那几只桶。验收看的是这几只，不是旁边已经认对的。
        miss_peak = 1.0
        for life, index, cx, cy in cold_tests:
            missed = frames[life][index]
            miss_heat = _heat_of(net, missed, device)
            positive = label_mask(missed, backgrounds[life], int(metas[life][index, 0]), int(metas[life][index, 1]), int(metas[life][index, 2]))
            if positive is None:
                miss_peak = 0.0
                continue
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(positive, 8)
            nearest = None
            nearest_gap = 1e9
            for comp in range(1, count):
                gap = abs(float(centroids[comp][0]) - cx) + abs(float(centroids[comp][1]) - cy)
                if gap < nearest_gap:
                    nearest_gap = gap
                    nearest = comp
            if nearest is None or nearest_gap > 30:
                miss_peak = 0.0
                continue
            ys, xs = np.where(labels == nearest)
            miss_peak = min(miss_peak, float(miss_heat[ys // BARREL_STRIDE, xs // BARREL_STRIDE].max()))
        print(
            f"{tag} wall {wall:.3f} miss {miss_peak:.3f} "
            f"left {left:.3f} right {right:.3f} chute {chute:.3f} round {whole:.3f} "
            f"body {body:.3f} arm {arm:.3f} edge {edge:.3f} pieces {pieces:.3f} kept {kept:.3f} broken {broken:.3f}",
            flush=True,
        )
        ok = (
            wall < 0.55 and miss_peak >= 0.7
            and left < 0.55 and right < 0.55 and chute >= 0.85 and whole >= 0.85
            and body < 0.45 and arm < 0.45 and edge < 0.55 and pieces < 0.55
            and kept >= 0.85 and broken < 0.55
        )
        return ok

    print("before", end=" ", flush=True)
    report("before")
    # 前面几层已经能把画在墙上的圆脸和真桶分成不同的特征。只改最后一层，避免整屏一起动。
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("net.12.")
    net.train()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=1e-3)
    order = np.arange(len(images))
    batch = 8
    pos_weight = torch.tensor([1.0], device=device)
    for epoch in range(80):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch):
            pick = order[start:start + batch]
            batch_x = np.stack([images[i] for i in pick]).astype(np.float32) / 255.0
            batch_y = np.stack([targets[i] for i in pick])
            batch_w = np.stack([weights[i] for i in pick])
            tensor_x = torch.from_numpy(np.ascontiguousarray(batch_x)).permute(0, 3, 1, 2).to(device)
            tensor_y = torch.from_numpy(np.ascontiguousarray(batch_y)).unsqueeze(1).to(device)
            tensor_w = torch.from_numpy(np.ascontiguousarray(batch_w)).unsqueeze(1).to(device)
            logits = net(tensor_x)
            if logits.shape[-2:] != tensor_y.shape[-2:]:
                logits = F.interpolate(logits, size=tensor_y.shape[-2:], mode="bilinear", align_corners=False)
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, pos_weight=pos_weight, reduction="none")
            denom = tensor_w.sum().clamp(min=1.0)
            loss = (loss_map * tensor_w).sum() / denom
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        net.eval()
        passed = report(f"epoch {epoch} loss {total / max(steps, 1):.4f}")
        left_now = region_peak(play_frame, 30, 140, 80, 200)
        pieces_now = piece_peak("/tmp/hitheight/edge_pieces.npz", backgrounds[0])
        if passed:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
        # 半片或人身上被抬起来就停。墙上的误报不能靠把碎片认成桶来换。
        if left_now > 0.45 or pieces_now > 0.4 or region_peak(round_frame, 50, 100, 120, 170) > 0.2:
            print("regression moved, stop", flush=True)
            break
        net.train()
    print("reject, keep old weights", flush=True)


def teach_bright():
    """两步认桶。先按亮度留下比空墙亮的元素，再从里面挑要打的整桶。

    碎块和人也可以比墙亮，那些格子标成不是目标。画在墙上、和墙一样亮的圆脸
    到不了第一步，不在这里当成桶去学。滑槽、圆脸、人身上、半片对不上就不写权重。
    """
    from main.barrel_vision import BRIGHT_COVER, _bright_cover

    frames, metas, backgrounds = load_lives()
    wall = np.median(np.stack(backgrounds), axis=0).astype(np.uint8)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    torch.save(blob, Path("/tmp/barrel_vision_before_bright.pt"))
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    threshold = float(blob["threshold"])
    play_frame = np.load("/tmp/hitheight/play_frags.npy")[1]
    round_frame = frames[0][250]
    chute_frame = frames[0][200]
    edge_probe = np.load("/tmp/hitheight/edge_pieces.npz")["frames"][0]
    keepers = (
        (frames[0][250], 120, 140, 200, 200),
        (frames[0][200], 90, 20, 180, 80),
        (frames[2][480], 210, 145, 256, 200),
        (frames[0][1072], 140, 145, 190, 200),
        (frames[2][1048], 60, 145, 120, 200),
    )

    def gated(frame):
        heat = _heat_of(net, frame, device)
        cover = _bright_cover(frame, wall, heat)
        kept = heat.copy()
        kept[cover < BRIGHT_COVER] = 0
        return kept

    def region_peak(image, x0, y0, x1, y1):
        heat = gated(image)
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    def piece_peak(pack_path):
        pack = np.load(pack_path)
        worst = 0.0
        saw = False
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = gated(frame)
            if hard.shape != heat.shape or not hard.any():
                continue
            saw = True
            worst = max(worst, float(heat[hard].max()))
        return worst if saw else 1.0

    def bright_barrel_cells(frame, background, agent_x, agent_y, status, heat):
        """有效元素里，够大、又不在人身上的整桶。格子要有一块比墙亮。"""
        found = np.zeros_like(heat, np.float32)
        positive = label_mask(frame, background, agent_x, agent_y, status)
        if positive is None:
            return found
        cover = _bright_cover(frame, background, heat)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(positive, 8)
        for index in range(1, count):
            # 拳头和半片没有这么大。贴在身边的整桶要留下，不能按身体方框整块丢掉。
            if int(stats[index, cv2.CC_STAT_AREA]) < 200:
                continue
            ys, xs = np.where(labels == index)
            cy, cx = ys // BARREL_STRIDE, xs // BARREL_STRIDE
            keep = cover[cy, cx] >= BRIGHT_COVER
            found[cy[keep], cx[keep]] = 1.0
        return found

    def holdout_peak():
        frame = frames[0][800]
        meta = metas[0][800]
        heat = gated(frame)
        positive = label_mask(frame, backgrounds[0], int(meta[0]), int(meta[1]), int(meta[2]))
        if positive is None:
            return 0.0
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(positive, 8)
        nearest, gap = None, 1e9
        for index in range(1, count):
            distance = abs(float(centroids[index][0]) - 183) + abs(float(centroids[index][1]) - 162)
            if distance < gap:
                gap, nearest = distance, index
        if nearest is None or int(stats[nearest, cv2.CC_STAT_AREA]) < 200:
            return 0.0
        cells = bright_barrel_cells(frame, wall, int(meta[0]), int(meta[1]), int(meta[2]), _heat_of(net, frame, device))
        ys, xs = np.where(labels == nearest)
        return float(heat[ys // BARREL_STRIDE, xs // BARREL_STRIDE][cells[ys // BARREL_STRIDE, xs // BARREL_STRIDE] > 0].max()) if (cells[ys // BARREL_STRIDE, xs // BARREL_STRIDE] > 0).any() else 0.0

    images, targets, weights = [], [], []
    for life in (0, 1):
        live, meta, background = frames[life], metas[life], backgrounds[life]
        for index in range(40, len(live), 8):
            if life == 0 and index == 800:
                continue
            agent_x, agent_y, status = int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2])
            heat = _heat_of(net, live[index], device)
            barrels = bright_barrel_cells(live[index], background, agent_x, agent_y, status, heat)
            pieces = _fragment_mask(live[index], background)
            piece_cells = np.zeros_like(heat, np.float32)
            if pieces is not None and int(pieces.sum()) >= 100:
                hard = _downsample(pieces)
                if hard.shape == heat.shape:
                    piece_cells = (hard > 0).astype(np.float32)
            if float(barrels.sum() + piece_cells.sum()) < 1:
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            low = (barrels > 0) & (heat < 0.7)
            target[low] = 1.0
            weight[low] = 8.0
            weight[piece_cells > 0] = 48.0
            if float(weight.sum()) < 1:
                continue
            copies = 4 if float(low.sum()) >= 1 else 1
            for _copy in range(copies):
                images.append(live[index])
                targets.append(target)
                weights.append(weight)
    for frame, x0, y0, x1, y1 in ((round_frame, 50, 100, 120, 170), (play_frame, 70, 110, 155, 175)):
        heat = _heat_of(net, frame, device)
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        weight[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] = 8.0
        images.append(frame)
        targets.append(target)
        weights.append(weight)
    for pack_path in ("/tmp/hitheight/edge_pieces.npz", "/tmp/hitheight/break_1103.npz"):
        pack = np.load(pack_path)
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            heat = _heat_of(net, frame, device)
            hard = _downsample(pieces)
            if hard.shape != heat.shape or not (hard > 0).any():
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            weight[hard > 0] = 64.0
            images.append(frame)
            targets.append(target)
            weights.append(weight)
    for frame, x0, y0, x1, y1 in keepers:
        heat = _heat_of(net, frame, device)
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        y_cells, x_cells = np.where(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] >= 0.85)
        target[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 1.0
        weight[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 4.0
        if float(weight.sum()) < 1:
            continue
        images.append(frame)
        targets.append(target)
        weights.append(weight)
    print(f"bright samples {len(images)} holdout {holdout_peak():.3f}", flush=True)

    def report(tag):
        miss = holdout_peak()
        chute = region_peak(chute_frame, 90, 10, 180, 80)
        whole = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        edge = region_peak(edge_probe, 180, 155, 240, 216)
        pieces = piece_peak("/tmp/hitheight/edge_pieces.npz")
        broken = piece_peak("/tmp/hitheight/break_1103.npz")
        kept = min(region_peak(image, x0, y0, x1, y1) for image, x0, y0, x1, y1 in keepers)
        wall_a = float(gated(frames[2][880])[106 // BARREL_STRIDE, 98 // BARREL_STRIDE])
        wall_b = float(gated(frames[2][1016])[142 // BARREL_STRIDE, 146 // BARREL_STRIDE])
        print(
            f"{tag} miss {miss:.3f} wall {wall_a:.3f}/{wall_b:.3f} "
            f"chute {chute:.3f} round {whole:.3f} body {body:.3f} arm {arm:.3f} "
            f"left {left:.3f} right {right:.3f} edge {edge:.3f} pieces {pieces:.3f} "
            f"kept {kept:.3f} broken {broken:.3f}",
            flush=True,
        )
        return (
            miss >= 0.7 and wall_a < 0.55 and wall_b < 0.55
            and chute >= 0.85 and whole >= 0.85 and body < 0.45 and arm < 0.45
            and left < 0.55 and right < 0.55 and edge < 0.55 and pieces < 0.55
            and kept >= 0.85 and broken < 0.55
        )

    report("before")
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("net.12.")
    net.train()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=5e-5)
    order = np.arange(len(images))
    for epoch in range(24):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), 8):
            pick = order[start:start + 8]
            batch_x = np.stack([images[i] for i in pick]).astype(np.float32) / 255.0
            batch_y = np.stack([targets[i] for i in pick])
            batch_w = np.stack([weights[i] for i in pick])
            tensor_x = torch.from_numpy(np.ascontiguousarray(batch_x)).permute(0, 3, 1, 2).to(device)
            tensor_y = torch.from_numpy(np.ascontiguousarray(batch_y)).unsqueeze(1).to(device)
            tensor_w = torch.from_numpy(np.ascontiguousarray(batch_w)).unsqueeze(1).to(device)
            logits = net(tensor_x)
            if logits.shape[-2:] != tensor_y.shape[-2:]:
                logits = F.interpolate(logits, size=tensor_y.shape[-2:], mode="bilinear", align_corners=False)
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, reduction="none")
            loss = (loss_map * tensor_w).sum() / tensor_w.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in net.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        net.eval()
        passed = report(f"epoch {epoch} loss {total / max(steps, 1):.4f}")
        if passed:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
                "background": wall,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
        if region_peak(play_frame, 30, 140, 80, 200) > 0.5 or piece_peak("/tmp/hitheight/edge_pieces.npz") > 0.5 or region_peak(round_frame, 50, 100, 120, 170) > 0.35:
            print("regression moved, stop", flush=True)
            break
        net.train()
    print("reject, keep old weights", flush=True)


# 拳头、脸、衣服的颜色。桶板上没有这些色，黑边不算。
_FIST_COLORS = np.array(
    [
        [232, 136, 64],
        [232, 168, 96],
        [232, 204, 0],
        [232, 136, 0],
        [168, 168, 168],
        [136, 136, 136],
        [96, 100, 96],
        [64, 68, 64],
        [32, 100, 232],
        [0, 68, 168],
        [232, 0, 0],
    ],
    np.int16,
)


def _fist_pixels(frame):
    """这一帧里拳头、脸和衣服占的像素。桶板的棕色不在这里面。"""
    flat = frame[:, :, :3].astype(np.int16).reshape(-1, 3)
    hit = np.zeros(flat.shape[0], dtype=bool)
    for color in _FIST_COLORS:
        hit |= np.all(flat == color, axis=1)
    return hit.reshape(frame.shape[:2])


def teach_fist():
    """用举起的拳头当反例。格子里大半是拳头或衣服、又没有桶板，才标成不是桶。

    贴在身边的整桶仍然是桶。滑槽、圆脸、碎片、人身上对不上就不写权重。
    """
    frames, metas, backgrounds = load_lives()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    torch.save(blob, Path("/tmp/barrel_vision_before_fist.pt"))
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    threshold = float(blob["threshold"])
    play_frame = np.load("/tmp/hitheight/play_frags.npy")[1]
    round_frame = frames[0][250]
    chute_frame = frames[0][200]
    edge_probe = np.load("/tmp/hitheight/edge_pieces.npz")["frames"][0]
    keepers = (
        (frames[0][250], 120, 140, 200, 200),
        (frames[0][200], 90, 20, 180, 80),
        (frames[2][480], 210, 145, 256, 200),
        (frames[0][1072], 140, 145, 190, 200),
        (frames[2][1048], 60, 145, 120, 200),
    )

    def region_peak(image, x0, y0, x1, y1):
        heat = _heat_of(net, image, device)
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    def piece_peak(pack_path):
        pack = np.load(pack_path)
        worst = 0.0
        saw = False
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = _heat_of(net, frame, device)
            if hard.shape != heat.shape or not hard.any():
                continue
            saw = True
            worst = max(worst, float(heat[hard].max()))
        return worst if saw else 1.0

    def fist_cells(picture, background, positive, heat, body):
        person = _fist_pixels(picture)
        reach = np.zeros(person.shape, dtype=bool)
        reach[60:190, max(0, body - 56):min(person.shape[1], body + 56)] = True
        person &= reach
        if positive is not None:
            person &= positive == 0
        cover = _block_mean(person.astype(np.float32), heat)
        barrel = np.zeros_like(heat) if positive is None else _block_mean(positive.astype(np.float32), heat)
        return (cover >= 0.35) & (barrel < 0.2)

    from main.env_utils import make_barrel_env

    images, targets, weights = [], [], []
    hold_images, hold_bodies, hold_life = [], [], []
    env = make_barrel_env(live_only=True, state_obs=True, seed=0, max_episode_steps=8000, render=False)()
    bonus = env.env
    try:
        for life in range(3):
            bonus.force_barrel_index = life
            env.reset(seed=0)
            done = False
            steps = 0
            while not done:
                frame = bonus.env.render()
                info = bonus.env.unwrapped.data.lookup_all()
                if frame is not None and steps % 2 == 0:
                    picture = np.ascontiguousarray(frame[:, :, :3])
                    ax = int(info.get("agent_x", 0))
                    body = 73 + (ax - 205)
                    positive = label_mask(
                        picture,
                        backgrounds[life],
                        ax,
                        int(info.get("agent_y", 192)),
                        int(info.get("agent_status", 0)),
                    )
                    heat = _heat_of(net, picture, device)
                    cells = fist_cells(picture, backgrounds[life], positive, heat, body)
                    peak = float(heat[cells].max()) if cells.any() else 0.0
                    # 热度已经越过门槛的拳头留一张平移 2 像素的检查，原图拿去学。
                    if peak >= 0.7:
                        shifted = np.roll(picture, 2, axis=1)
                        hold_images.append(shifted)
                        hold_bodies.append(body + 2)
                        hold_life.append(life)
                    if cells.any() and peak >= 0.2:
                        target = np.zeros_like(heat, np.float32)
                        weight = np.zeros_like(heat, np.float32)
                        # 已经认成桶的拳头多压几次。只是有点像的，轻轻压下去。
                        weight[cells & (heat >= 0.7)] = 40.0
                        weight[cells & (heat >= 0.4) & (heat < 0.7)] = 12.0
                        weight[cells & (heat >= 0.15) & (heat < 0.4)] = 6.0
                        if positive is not None:
                            barrel = _block_mean(positive.astype(np.float32), heat) >= 0.5
                            target[barrel & (heat >= 0.85)] = 1.0
                            weight[barrel & (heat >= 0.85)] = 4.0
                        if float(weight.sum()) >= 1:
                            copies = 8 if peak >= 0.7 else 1
                            for _copy in range(copies):
                                images.append(picture)
                                targets.append(target)
                                weights.append(weight)
                _, _, term, trunc, _ = env.step(int(bonus.rule_action()))
                done = bool(term or trunc)
                steps += 1
    finally:
        env.close()
    for frame, x0, y0, x1, y1 in keepers:
        heat = _heat_of(net, frame, device)
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        window = heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE] >= 0.85
        y_cells, x_cells = np.where(window)
        target[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 1.0
        weight[y_cells + y0 // BARREL_STRIDE, x_cells + x0 // BARREL_STRIDE] = 4.0
        if float(weight.sum()) < 1:
            continue
        images.append(frame)
        targets.append(target)
        weights.append(weight)
    for pack_path in ("/tmp/hitheight/edge_pieces.npz", "/tmp/hitheight/break_1103.npz"):
        pack = np.load(pack_path)
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            heat = _heat_of(net, frame, device)
            hard = _downsample(pieces)
            if hard.shape != heat.shape or not (hard > 0).any():
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            weight[hard > 0] = 8.0
            images.append(frame)
            targets.append(target)
            weights.append(weight)

    def holdout_peak():
        worst = 0.0
        for picture, body, life in zip(hold_images, hold_bodies, hold_life):
            heat = _heat_of(net, picture, device)
            # 检查图是平移过的，不能再和原来的空墙相减。颜色本身就够指出拳头。
            cells = fist_cells(picture, backgrounds[life], None, heat, body)
            if cells.any():
                worst = max(worst, float(heat[cells].max()))
        return worst

    print(f"fist samples {len(images)} hold {len(hold_images)} holdout {holdout_peak():.3f}", flush=True)

    def report(tag):
        fist = holdout_peak()
        chute = region_peak(chute_frame, 90, 10, 180, 80)
        whole = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        edge = region_peak(edge_probe, 180, 155, 240, 216)
        pieces = piece_peak("/tmp/hitheight/edge_pieces.npz")
        broken = piece_peak("/tmp/hitheight/break_1103.npz")
        kept = min(region_peak(image, x0, y0, x1, y1) for image, x0, y0, x1, y1 in keepers)
        print(
            f"{tag} fist {fist:.3f} chute {chute:.3f} round {whole:.3f} "
            f"body {body:.3f} arm {arm:.3f} left {left:.3f} right {right:.3f} "
            f"edge {edge:.3f} pieces {pieces:.3f} kept {kept:.3f} broken {broken:.3f}",
            flush=True,
        )
        return (
            fist < 0.45
            and chute >= 0.85 and whole >= 0.85 and body < 0.45 and arm < 0.45
            and left < 0.55 and right < 0.55 and edge < 0.55 and pieces < 0.55
            and kept >= 0.85 and broken < 0.55
        )

    report("before")
    if not images:
        print("reject, keep old weights", flush=True)
        return
    # 一格的感受野只够一只桶，拳头贴在桶旁边时这一格仍像桶。只训练周围那一圈，从零开始。
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("wide.")
    net.train()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=1e-4)
    order = np.arange(len(images))
    best_fist = 1.0
    stalled = 0
    for epoch in range(48):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps_done = 0
        for start in range(0, len(order), 8):
            pick = order[start:start + 8]
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
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, reduction="none")
            loss = (loss_map * tensor_w).sum() / tensor_w.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in net.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps_done += 1
        net.eval()
        passed = report(f"epoch {epoch} loss {total / max(steps_done, 1):.4f}")
        fist_now = holdout_peak()
        if fist_now < best_fist - 0.002:
            best_fist = fist_now
            stalled = 0
        else:
            stalled += 1
        if passed:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
        if stalled >= 8:
            print("fist stopped moving, stop", flush=True)
            break
        if (
            region_peak(play_frame, 30, 140, 80, 200) > 0.5
            or piece_peak("/tmp/hitheight/edge_pieces.npz") > 0.45
            or region_peak(round_frame, 50, 100, 120, 170) > 0.35
            or region_peak(chute_frame, 90, 10, 180, 80) < 0.8
            or region_peak(round_frame, 120, 140, 200, 200) < 0.8
        ):
            print("regression moved, stop", flush=True)
            break
        net.train()
    print("reject, keep old weights", flush=True)


def _whole_and_chute(positive):
    """完整的桶大约四十像素见方、又实。滑槽里是扁的侧面，不拿来当目标。"""
    whole = np.zeros_like(positive)
    chute = np.zeros_like(positive)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(positive, 8)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        cy = float(centroids[index][1])
        component = (labels == index).astype(np.uint8)
        if 36 <= width <= 52 and 36 <= height <= 52 and area >= 700 and _solidity(component) >= 0.85:
            whole[component > 0] = 1
        elif height <= 32 and width >= 28 and cy < 90 and area >= 200:
            chute[component > 0] = 1
    return whole, chute


def teach_whole():
    """只把完整的桶当成要认的目标，滑槽里的侧面标成不是桶。

    完整桶的置信度要留在门槛上，滑槽掉下去。碎片和人身上冒出来就不写权重。
    """
    frames, metas, backgrounds = load_lives()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    torch.save(blob, Path("/tmp/barrel_vision_before_whole.pt"))
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    threshold = float(blob["threshold"])
    play_frame = np.load("/tmp/hitheight/play_frags.npy")[1]
    round_frame = frames[0][250]
    chute_frame = frames[0][200]
    edge_probe = np.load("/tmp/hitheight/edge_pieces.npz")["frames"][0]
    images, targets, weights = [], [], []
    hold_whole, hold_chute = [], []

    def cells_of(mask, heat):
        down = _downsample(mask)
        if down.shape != heat.shape:
            return np.zeros_like(heat, dtype=bool)
        return down > 0

    for life in (0, 1, 2):
        live, meta, background = frames[life], metas[life], backgrounds[life]
        for index in range(40, len(live), 8):
            positive = label_mask(live[index], background, int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
            if positive is None:
                continue
            whole, chute = _whole_and_chute(positive)
            if int(whole.sum()) < 100 and int(chute.sum()) < 100:
                continue
            heat = _heat_of(net, live[index], device)
            whole_cells = cells_of(whole, heat)
            chute_cells = cells_of(chute, heat) & ~whole_cells
            # 留几张完整桶和滑槽侧面做检查，不拿去学。
            if life == 2 and index % 40 == 0:
                pose = (live[index], int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
                if whole_cells.any():
                    hold_whole.append(pose)
                if chute_cells.any():
                    hold_chute.append(pose)
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            target[whole_cells] = 1.0
            weight[whole_cells & (heat < 0.95)] = 12.0
            weight[whole_cells & (heat >= 0.95)] = 8.0
            weight[chute_cells] = 16.0
            if float(weight.sum()) < 1:
                continue
            images.append(live[index])
            targets.append(target)
            weights.append(weight)
    for pack_path in ("/tmp/hitheight/edge_pieces.npz", "/tmp/hitheight/break_1103.npz"):
        pack = np.load(pack_path)
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            heat = _heat_of(net, frame, device)
            hard = _downsample(pieces)
            if hard.shape != heat.shape or not (hard > 0).any():
                continue
            target = np.zeros_like(heat, np.float32)
            weight = np.zeros_like(heat, np.float32)
            weight[hard > 0] = 8.0
            images.append(frame)
            targets.append(target)
            weights.append(weight)

    def region_peak(image, x0, y0, x1, y1):
        heat = _heat_of(net, image, device)
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    def piece_peak(pack_path):
        pack = np.load(pack_path)
        worst = 0.0
        saw = False
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = _heat_of(net, frame, device)
            if hard.shape != heat.shape or not hard.any():
                continue
            saw = True
            worst = max(worst, float(heat[hard].max()))
        return worst if saw else 1.0

    def whole_peak(samples):
        worst = 1.0
        for frame, agent_x, agent_y, status in samples:
            positive = label_mask(frame, backgrounds[2], agent_x, agent_y, status)
            if positive is None:
                continue
            whole, _chute = _whole_and_chute(positive)
            heat = _heat_of(net, frame, device)
            cells = cells_of(whole, heat)
            if cells.any():
                worst = min(worst, float(heat[cells].max()))
        return worst

    def chute_peak(samples):
        worst = 0.0
        for frame, agent_x, agent_y, status in samples:
            positive = label_mask(frame, backgrounds[2], agent_x, agent_y, status)
            if positive is None:
                continue
            _whole, chute = _whole_and_chute(positive)
            heat = _heat_of(net, frame, device)
            cells = cells_of(chute, heat)
            if cells.any():
                worst = max(worst, float(heat[cells].max()))
        return worst

    print(f"whole samples {len(images)} hold whole {len(hold_whole)} chute {len(hold_chute)}", flush=True)

    def report(tag):
        whole = whole_peak(hold_whole)
        chute = chute_peak(hold_chute)
        ground = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        edge = region_peak(edge_probe, 180, 155, 240, 216)
        pieces = piece_peak("/tmp/hitheight/edge_pieces.npz")
        broken = piece_peak("/tmp/hitheight/break_1103.npz")
        print(
            f"{tag} whole {whole:.3f} chute {chute:.3f} round {ground:.3f} "
            f"body {body:.3f} arm {arm:.3f} left {left:.3f} right {right:.3f} "
            f"edge {edge:.3f} pieces {pieces:.3f} broken {broken:.3f}",
            flush=True,
        )
        return (
            whole >= 0.95 and ground >= 0.95 and chute <= 0.55
            and body < 0.45 and arm < 0.45 and left < 0.55 and right < 0.55
            and edge < 0.55 and pieces < 0.55 and broken < 0.55
        )

    report("before")
    if not images or not hold_whole:
        print("reject, keep old weights", flush=True)
        return
    # 最后一层看到的特征里，滑槽侧面和完整桶是同一类。放开前一层 3x3，步子放很小。
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("net.10.") or name.startswith("net.12.")
    net.train()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=3e-6)
    order = np.arange(len(images))
    best_chute = 1.0
    stalled = 0
    for epoch in range(24):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps_done = 0
        for start in range(0, len(order), 8):
            pick = order[start:start + 8]
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
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, reduction="none")
            loss = (loss_map * tensor_w).sum() / tensor_w.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in net.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps_done += 1
        net.eval()
        passed = report(f"epoch {epoch} loss {total / max(steps_done, 1):.4f}")
        chute_now = chute_peak(hold_chute)
        if chute_now < best_chute - 0.005:
            best_chute = chute_now
            stalled = 0
        else:
            stalled += 1
        if passed:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
        if stalled >= 8:
            print("chute stopped moving, stop", flush=True)
            break
        if whole_peak(hold_whole) < 0.96 or region_peak(round_frame, 120, 140, 200, 200) < 0.96 or piece_peak("/tmp/hitheight/edge_pieces.npz") > 0.3 or region_peak(round_frame, 50, 100, 120, 170) > 0.3:
            print("regression moved, stop", flush=True)
            break
        net.train()
    print("reject, keep old weights", flush=True)


def _wall_pixels(frame, background, whole):
    """和空墙一样的像素。画在墙上的圆脸也算墙。完整桶盖住的地方不算。"""
    diff = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2) < 30
    if whole is not None:
        cover = cv2.dilate(whole, np.ones((3, 3), np.uint8))
        diff[cover > 0] = False
    diff[:, 256:] = False
    diff[:16, :] = False
    return diff


# 波动拳飞出去之后的火球。打在桶上的火花不在这里，那一块本来就有桶。
_HADOU_COLORS = np.array(
    [
        [232, 236, 232],
        [64, 204, 232],
        [0, 100, 232],
        [0, 32, 200],
        [32, 168, 200],
    ],
    np.int16,
)


def _hadou_pixels(frame, body):
    """离开身体的波动拳。贴在拳头上的那一团不算，打中桶时那里仍是桶。"""
    flat = frame[:, :, :3].astype(np.int16).reshape(-1, 3)
    hit = np.zeros(flat.shape[0], dtype=bool)
    for color in _HADOU_COLORS:
        hit |= np.all(flat == color, axis=1)
    mask = hit.reshape(frame.shape[:2])
    mask[:, 256:] = False
    mask[:16, :] = False
    # 人自己的头带也有接近的蓝色，身前这一小段留给拳头。
    x0 = max(0, int(body) - 24)
    x1 = min(mask.shape[1], int(body) + 28)
    mask[90:175, x0:x1] = False
    return mask


def teach_cases():
    """完整的桶是目标。反例是背景墙、波动拳的火球，和半片破口上的锯齿。

    方框只盖破损的那一条，圆边留在框外。完整桶或滑槽掉了就不写权重。
    """
    frames, metas, backgrounds = load_lives()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    torch.save(blob, Path("/tmp/barrel_vision_before_cases.pt"))
    net = BarrelNet().to(device)
    net.load_state_dict(blob["state"], strict=False)
    net.eval()
    threshold = float(blob["threshold"])
    play_frame = np.load("/tmp/hitheight/play_frags.npy")[1]
    round_frame = frames[0][250]
    edge_probe = np.load("/tmp/hitheight/edge_pieces.npz")["frames"][0]
    images, targets, weights = [], [], []
    hold_whole = []
    hold_fist = []
    hold_wall = []
    hold_piece = []

    def cells_of(mask, heat):
        if mask is None or int(np.asarray(mask).sum()) < 1:
            return np.zeros(heat.shape, dtype=bool)
        down = _downsample(mask.astype(np.uint8))
        if down.shape != heat.shape:
            return np.zeros(heat.shape, dtype=bool)
        return down > 0

    for life in (0, 1, 2):
        live, meta, background = frames[life], metas[life], backgrounds[life]
        previous = None
        for index in range(40, len(live), 6):
            agent_x = int(meta[index, 0])
            agent_y = int(meta[index, 1])
            status = int(meta[index, 2])
            body = 73 + (agent_x - 205)
            positive = label_mask(live[index], background, agent_x, agent_y, status)
            if positive is None:
                previous = live[index]
                continue
            whole, _chute = _whole_and_chute(positive)
            heat = _heat_of(net, live[index], device)
            whole_cells = cells_of(whole, heat)
            wall = _wall_pixels(live[index], background, whole)
            fire = _hadou_pixels(live[index], body)
            fire[whole > 0] = False
            wall_cells = cells_of(wall, heat) & ~whole_cells & (heat >= 0.2)
            fire_cells = cells_of(fire, heat) & ~whole_cells
            fragments = _ground_fragments(live[index], background, whole)
            fragment_cells = cells_of(fragments, heat)
            pose_frame = live[index]
            # 留一帧贴地半片不参与学习，用来看有没有真的压下去。
            if life == 2 and index == 628 and fragment_cells.any():
                hold_piece.append(pose_frame)
            elif life == 2 and (index - 40) % 42 == 0 and whole_cells.any():
                hold_whole.append((pose_frame, agent_x, agent_y, status))
            elif life == 1 and (index - 40) % 42 == 0 and wall_cells.any():
                hold_wall.append((pose_frame, agent_x, agent_y, status))
            else:
                target = np.zeros_like(heat, np.float32)
                weight = np.zeros_like(heat, np.float32)
                target[whole_cells] = 1.0
                weight[whole_cells & (heat >= 0.9)] = 8.0
                weight[whole_cells & (heat < 0.9)] = 12.0
                weight[fire_cells & (heat >= 0.15)] = 8.0
                # 反例只放在破口的方框上。整片桶板和圆边不跟着记成反例。
                jagged_cells = cells_of(_box_mask(pose_frame.shape, _jagged_boxes(pose_frame, fragments)), heat)
                weight[jagged_cells] = 24.0
                copies = 4 if jagged_cells.any() else 1
                if float(weight.sum()) >= 1:
                    for _repeat in range(copies):
                        images.append(pose_frame)
                        targets.append(target)
                        weights.append(weight)
            previous = live[index]

    extra_frames = []
    edge_pack = np.load("/tmp/hitheight/edge_pieces.npz")
    for frame, life in zip(edge_pack["frames"], edge_pack["life"]):
        extra_frames.append((frame, backgrounds[int(life)]))
    shard_frame = cv2.cvtColor(cv2.imread("/tmp/barrel_shots/hadou_30.21.png"), cv2.COLOR_BGR2RGB)
    extra_frames.append((shard_frame, backgrounds[0]))
    for frame, background in extra_frames:
        whole = np.zeros(frame.shape[:2], np.uint8)
        fragments = _ground_fragments(frame, background, whole)
        if int(fragments.sum()) < 80:
            continue
        heat = _heat_of(net, frame, device)
        fragment_cells = cells_of(_box_mask(frame.shape, _jagged_boxes(frame, fragments)), heat)
        if not fragment_cells.any():
            continue
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        weight[fragment_cells] = 24.0
        for _repeat in range(6):
            images.append(frame)
            targets.append(target)
            weights.append(weight)

    # 确认过的样本每次都加进去。不换墙上的桶，不换火球，左右半片都用方框。
    locked = _locked_spec()
    locked_items = [
        (locked["positive"], 1.0, 12.0),
        (locked["wall"], 0.0, 10.0),
        (locked["fireball"], 0.0, 10.0),
    ]
    for item, goal, gain in locked_items:
        frame = _locked_frame(item["file"])
        heat = _heat_of(net, frame, device)
        cells = cells_of(_box_mask(frame.shape, item["boxes"]), heat)
        if not cells.any():
            continue
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        target[cells] = goal
        weight[cells] = gain
        for _repeat in range(8):
            images.append(frame)
            targets.append(target)
            weights.append(weight)
    # 锁定的半片只把破口那一条放进反例。整片的框留着，用来核对有没有压下去。
    for item in locked["fragments"]:
        frame = _locked_frame(item["file"])
        jagged = [box for box in (_jagged_box(frame, box) for box in item["boxes"]) if box is not None]
        print(f"jagged {item['file']} {jagged}", flush=True)
        heat = _heat_of(net, frame, device)
        cells = cells_of(_box_mask(frame.shape, jagged), heat)
        if not cells.any():
            continue
        target = np.zeros_like(heat, np.float32)
        weight = np.zeros_like(heat, np.float32)
        weight[cells] = 24.0
        for _repeat in range(8):
            images.append(frame)
            targets.append(target)
            weights.append(weight)

    def region_peak(image, x0, y0, x1, y1):
        heat = _heat_of(net, image, device)
        return float(heat[y0 // BARREL_STRIDE:y1 // BARREL_STRIDE, x0 // BARREL_STRIDE:x1 // BARREL_STRIDE].max())

    def piece_peak(pack_path):
        pack = np.load(pack_path)
        worst = 0.0
        saw = False
        lives = pack["life"] if "life" in pack.files else np.zeros(len(pack["frames"]), np.int32)
        for frame, life in zip(pack["frames"], lives):
            pieces = _fragment_mask(frame, backgrounds[int(life)])
            if pieces is None or int(pieces.sum()) < 100:
                continue
            hard = _downsample(pieces) > 0
            heat = _heat_of(net, frame, device)
            if hard.shape != heat.shape or not hard.any():
                continue
            saw = True
            worst = max(worst, float(heat[hard].max()))
        return worst if saw else 1.0

    def shard_peak(frame, background):
        pieces = _ground_fragments(frame, background)
        heat = _heat_of(net, frame, device)
        cells = cells_of(pieces, heat)
        if not cells.any():
            return 0.0
        return float(heat[cells].max())

    def whole_peak():
        worst = 1.0
        for frame, agent_x, agent_y, status in hold_whole:
            positive = label_mask(frame, backgrounds[2], agent_x, agent_y, status)
            if positive is None:
                continue
            whole, _chute = _whole_and_chute(positive)
            heat = _heat_of(net, frame, device)
            cells = cells_of(whole, heat)
            if cells.any():
                worst = min(worst, float(heat[cells].max()))
        return worst

    def wall_peak():
        worst = 0.0
        for frame, agent_x, agent_y, status in hold_wall:
            heat = _heat_of(net, frame, device)
            positive = label_mask(frame, backgrounds[1], agent_x, agent_y, status)
            whole = np.zeros(frame.shape[:2], np.uint8) if positive is None else _whole_and_chute(positive)[0]
            cells = cells_of(_wall_pixels(frame, backgrounds[1], whole), heat)
            hot = cells & (heat >= 0.2)
            if hot.any():
                worst = max(worst, float(heat[hot].max()))
        return worst

    def fist_peak():
        worst = 0.0
        for frame, body in hold_fist:
            heat = _heat_of(net, frame, device)
            person = _fist_pixels(frame)
            reach = np.zeros(person.shape, dtype=bool)
            reach[60:190, max(0, int(body) - 56):min(person.shape[1], int(body) + 56)] = True
            cells = cells_of(person & reach, heat)
            if cells.any():
                worst = max(worst, float(heat[cells].max()))
        return worst

    print(
        f"case samples {len(images)} hold whole {len(hold_whole)} wall {len(hold_wall)} fist {len(hold_fist)}",
        flush=True,
    )

    def report(tag):
        whole = whole_peak()
        wall = wall_peak()
        ground = region_peak(round_frame, 120, 140, 200, 200)
        body = region_peak(round_frame, 50, 100, 120, 170)
        arm = region_peak(play_frame, 70, 110, 155, 175)
        left = region_peak(play_frame, 30, 140, 80, 200)
        right = region_peak(play_frame, 180, 140, 240, 200)
        edge = region_peak(edge_probe, 180, 155, 240, 216)
        pieces = piece_peak("/tmp/hitheight/edge_pieces.npz")
        broken = piece_peak("/tmp/hitheight/break_1103.npz")
        chute = region_peak(frames[0][200], 90, 10, 180, 80)
        shard = shard_peak(shard_frame, backgrounds[0])
        held = max((shard_peak(frame, backgrounds[2]) for frame in hold_piece), default=0.0)

        def box_peak(item):
            frame = _locked_frame(item["file"])
            heat = _heat_of(net, frame, device)
            cells = cells_of(_box_mask(frame.shape, item["boxes"]), heat)
            if not cells.any():
                return 0.0
            return float(heat[cells].max())

        locked_pos = box_peak(locked["positive"])
        locked_wall = box_peak(locked["wall"])
        locked_fire = box_peak(locked["fireball"])
        locked_frag = max(box_peak(item) for item in locked["fragments"])
        print(
            f"{tag} whole {whole:.3f} round {ground:.3f} chute {chute:.3f} wall {wall:.3f} "
            f"body {body:.3f} arm {arm:.3f} left {left:.3f} right {right:.3f} "
            f"edge {edge:.3f} pieces {pieces:.3f} broken {broken:.3f} shard {shard:.3f} held {held:.3f} "
            f"lock pos {locked_pos:.3f} wall {locked_wall:.3f} fire {locked_fire:.3f} frag {locked_frag:.3f}",
            flush=True,
        )
        return (
            whole >= 0.95 and ground >= 0.95 and chute >= 0.85
            and body < 0.45 and arm < 0.45 and left < 0.55 and right < 0.55
            and edge < 0.55 and pieces < 0.55 and broken < 0.55
            and shard < 0.55 and held < 0.55
            and locked_pos >= 0.95 and locked_wall < 0.55 and locked_fire < 0.55 and locked_frag < 0.55
        )

    report("before")
    if not images or not hold_whole:
        print("reject, keep old weights", flush=True)
        return
    for name, param in net.named_parameters():
        # 半片和整桶在最后一层上长得一样。只学多看一圈的那层，让小半片掉下去，整桶的原输出留着。
        param.requires_grad = name.startswith("wide.")
    net.train()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=1e-4)
    order = np.arange(len(images))
    best_bad = 1.0
    stalled = 0
    for epoch in range(24):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        total = 0.0
        steps_done = 0
        for start in range(0, len(order), 8):
            pick = order[start:start + 8]
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
            loss_map = F.binary_cross_entropy_with_logits(logits, tensor_y, reduction="none")
            loss = (loss_map * tensor_w).sum() / tensor_w.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in net.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps_done += 1
        net.eval()
        passed = report(f"epoch {epoch} loss {total / max(steps_done, 1):.4f}")
        bad = shard_peak(shard_frame, backgrounds[0])
        if bad < best_bad - 0.01:
            best_bad = bad
            stalled = 0
        else:
            stalled += 1
        if passed:
            trained = {
                "state": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "threshold": threshold,
            }
            torch.save(trained, WEIGHT_PATH)
            print(f"wrote {WEIGHT_PATH}", flush=True)
            return
        if stalled >= 8:
            print("negatives stopped moving, stop", flush=True)
            break
        if whole_peak() < 0.96 or region_peak(round_frame, 120, 140, 200, 200) < 0.96 or region_peak(frames[0][200], 90, 10, 180, 80) < 0.85 or region_peak(round_frame, 50, 100, 120, 170) > 0.35 or piece_peak("/tmp/hitheight/edge_pieces.npz") > 0.45 or region_peak(play_frame, 180, 140, 240, 200) > 0.5:
            print("whole barrel moved, stop", flush=True)
            break
        net.train()
    print("reject, keep old weights", flush=True)


def teach_crop():
    """热度点裁成原像素的一块，学这一块是不是整桶。

    圆脸和滑槽是桶。半片和火球不是。墙上的圆第一段已经不报，不放进这块。整桶掉了或半片还在门槛上就不写权重。
    """
    from main.barrel_vision import BarrelCropNet, CROP_PATH, crop_at

    frames, metas, backgrounds = load_lives()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    stage1 = BarrelNet().to(device)
    blob = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    stage1.load_state_dict(blob["state"], strict=False)
    stage1.eval()

    def parts(frame, background, agent_x, agent_y, status):
        positive = label_mask(frame, background, agent_x, agent_y, status)
        if positive is None:
            return None, None, _ground_fragments(frame, background, None)
        whole, chute = _whole_and_chute(positive)
        return whole, chute, _ground_fragments(frame, background, whole)

    def centers(mask):
        if mask is None or int(mask.sum()) < 1:
            return []
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        found = []
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < 80:
                continue
            found.append((float(centroids[index][0]), float(centroids[index][1])))
        return found

    train, evals = [], {name: [] for name in ("whole", "chute", "frag", "wall", "fire", "body", "locked", "positive")}
    for life in (0, 1, 2):
        live, meta, background = frames[life], metas[life], backgrounds[life]
        for index in range(40, len(live), 8):
            agent_x, agent_y, status = (int(meta[index, 0]), int(meta[index, 1]), int(meta[index, 2]))
            whole, chute, frags = parts(live[index], background, agent_x, agent_y, status)
            held = life == 2 and index % 16 == 0
            buckets = (
                ("whole", whole, 1.0),
                ("chute", chute, 1.0),
                ("frag", frags, 0.0),
            )
            for kind, mask, goal in buckets:
                for cx, cy in centers(mask):
                    image = crop_at(live[index], cx, cy)
                    if (kind == "frag" and life == 2) or held:
                        evals[kind].append(image)
                        continue
                    train.append((image, goal, 6.0 if kind == "frag" else 1.0))
                    for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8)):
                        train.append((crop_at(live[index], cx + dx, cy + dy), goal, 1.0))
            # 墙上的圆第一段热度已经是 0，不会进裁块。裁块里它和真桶太像，不拿来当反例。
            wall = (_brown(live[index]) & (np.abs(live[index].astype(np.int16) - background.astype(np.int16)).sum(axis=2) < 30)).astype(np.uint8)
            wall[:, 256:] = 0
            if held:
                for cx, cy, _w, _h, _area in _component_boxes(wall):
                    if 30 <= _w <= 55 and 20 <= _h <= 55:
                        evals["wall"].append(crop_at(live[index], cx, cy))
            heat = _heat_of(stage1, live[index], device)
            for cx, cy, _w, _h, _area in _component_boxes((heat >= 0.7).astype(np.uint8)):
                px = cx * BARREL_STRIDE + BARREL_STRIDE / 2
                py = cy * BARREL_STRIDE + BARREL_STRIDE / 2
                iy = int(np.clip(py, 0, live[index].shape[0] - 1))
                ix = int(np.clip(px, 0, live[index].shape[1] - 1))
                if whole is not None and whole[iy, ix]:
                    goal, kind = 1.0, "whole"
                elif chute is not None and chute[iy, ix]:
                    goal, kind = 1.0, "chute"
                elif frags[iy, ix]:
                    goal, kind = 0.0, "frag"
                else:
                    continue
                image = crop_at(live[index], px, py)
                if (kind == "frag" and life == 2) or held:
                    evals[kind].append(image)
                    continue
                train.append((image, goal, 6.0 if kind == "frag" else 1.0))

    locked = _locked_spec()
    positive = _locked_frame(locked["positive"]["file"])
    box = locked["positive"]["boxes"][0]
    evals["positive"].append(crop_at(positive, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    for key in ("wall", "fireball"):
        frame = _locked_frame(locked[key]["file"])
        box = locked[key]["boxes"][0]
        image = crop_at(frame, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        name = "fire" if key == "fireball" else "wall"
        evals[name].append(image)
        if key == "fireball":
            for _repeat in range(8):
                train.append((image, 0.0, 4.0))
    for item in locked["fragments"]:
        frame = _locked_frame(item["file"])
        for box in item["boxes"]:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            image = crop_at(frame, cx, cy)
            evals["locked"].append(image)
            for _repeat in range(12):
                train.append((image, 0.0, 8.0))
                for dx, dy in ((6, 0), (-6, 0), (0, 6), (0, -6)):
                    train.append((crop_at(frame, cx + dx, cy + dy), 0.0, 4.0))

    print(
        f"crop train {len(train)} eval whole {len(evals['whole'])} chute {len(evals['chute'])} "
        f"frag {len(evals['frag'])} wall {len(evals['wall'])} locked {len(evals['locked'])}",
        flush=True,
    )
    net = BarrelCropNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    order = np.arange(len(train))
    best = None
    best_gap = -1.0

    def measure(rows):
        if not rows:
            return np.zeros(0, np.float32)
        net.eval()
        found = []
        with torch.no_grad():
            for start in range(0, len(rows), 64):
                batch = np.stack(rows[start:start + 64]).astype(np.float32) / 255.0
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                found.append(torch.sigmoid(net(tensor)).cpu().numpy())
        return np.concatenate(found)

    for epoch in range(24):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        net.train()
        total = 0.0
        steps = 0
        for start in range(0, len(order), 32):
            pick = order[start:start + 32]
            batch = np.stack([train[i][0] for i in pick]).astype(np.float32) / 255.0
            target = torch.tensor([train[i][1] for i in pick], dtype=torch.float32, device=device)
            weight = torch.tensor([train[i][2] for i in pick], dtype=torch.float32, device=device)
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            if rng.random() < 0.5:
                tensor = torch.flip(tensor, dims=[3])
            logit = net(tensor)
            loss = (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        scores = {name: measure(rows) for name, rows in evals.items()}
        def stat(name, empty):
            values = scores[name]
            return empty if len(values) == 0 else float(values.min() if name in ("whole", "chute", "positive") else values.max())
        whole = stat("whole", 0.0)
        chute = stat("chute", 0.0)
        frag = stat("frag", 0.0)
        wall = stat("wall", 1.0)
        fire = stat("fire", 1.0)
        body = stat("body", 1.0)
        locked_frag = stat("locked", 1.0)
        positive = stat("positive", 0.0)
        gap = min(whole, chute, positive) - max(frag, fire, locked_frag)
        print(
            f"epoch {epoch} loss {total / max(steps, 1):.3f} whole {whole:.3f} chute {chute:.3f} "
            f"pos {positive:.3f} frag {frag:.3f} locked {locked_frag:.3f} wall {wall:.3f} "
            f"fire {fire:.3f} body {body:.3f}",
            flush=True,
        )
        passed = whole >= 0.85 and chute >= 0.85 and positive >= 0.85 and frag < 0.35 and locked_frag < 0.35 and fire < 0.45
        if passed and gap > best_gap:
            best_gap = gap
            best = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    if best is None:
        print("reject, crop model not written", flush=True)
        return
    torch.save({"state": best, "threshold": 0.7, "crop": 72}, CROP_PATH)
    print(f"wrote {CROP_PATH}", flush=True)


def _component_boxes(mask):
    """连通块的中心和宽高。给墙上的圆和碎片取裁块中心。"""
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    found = []
    for index in range(1, count):
        found.append((
            float(centroids[index][0]),
            float(centroids[index][1]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
            int(stats[index, cv2.CC_STAT_AREA]),
        ))
    return found


if __name__ == "__main__":
    train()
