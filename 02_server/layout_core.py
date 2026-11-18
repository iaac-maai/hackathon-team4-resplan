"""layout_core.py - Team 4 layout generator: the code shared by the notebooks and the model server.

Step 2  the model (LayoutCVAE) and how it invents room maps.
Step 3  from the model's picture (8 layers, 128 x 128) to a clean vector plan:
        rooms with straight walls, doors, windows, a score, and a DXF file.

Step 3 needs only numpy, OpenCV, SciPy and scikit-learn (ezdxf for the DXF file).
PyTorch is needed only to run the model.
"""
from collections import Counter

import numpy as np
import cv2
from scipy import ndimage
from sklearn.tree import DecisionTreeClassifier

try:
    import torch
    from torch import nn
except ImportError:          # step 3 works without PyTorch
    torch = None

IMG = 128
ROOMS = ["living", "bedroom", "bathroom", "kitchen", "balcony"]
CHANNELS = ROOMS + ["wall", "door", "window"]
BED_MAX = 5
COLORS = {"living": "#2a78d6", "bedroom": "#eb6834", "bathroom": "#1baf7a", "kitchen": "#eda100",
          "balcony": "#e87ba4", "wall": "#3d3d3a", "window": "#9ec5f4", "door": "#ffffff", "entrance": "#0b0b0b"}

# --------------------------------------------------------------------------------------------
# Settings for step 3 (all lengths in metres). Change them here, or pass your own dictionary.
# --------------------------------------------------------------------------------------------
SETTINGS = {
    "exterior_wall_m": 0.20,
    "interior_wall_m": 0.10,
    "door_m": {"bedroom": 0.80, "bathroom": 0.70, "kitchen": 1.20, "balcony": 0.90, "living": 1.20},
    "entrance_m": 0.95,
    "window_m": {"living": 1.80, "bedroom": 1.40, "kitchen": 1.20, "bathroom": 0.60},
    "min_bedroom_m2": 9.0,         # net, inside the walls: a smaller bedroom and the option is dropped
    "min_bedroom_width_m": 2.5,    # net, inside the walls
    "min_main_bathroom_m2": 3.5,   # net: at least one bathroom this big (shower, toilet, sink); smaller ones are guest toilets
    "kitchen_boost": [1.0, 1.5, 2.0, 3.0],   # no kitchen in the picture: the model's kitchen signal is made stronger, step by step
    "kitchen_outer_wall_m": 1.0,   # the kitchen touches the outside along at least 1 m (97% of ResPlan kitchens do)
    "max_corners": {"bedroom": 6, "bathroom": 6, "kitchen": 6, "balcony": 6, "living": 12},   # 4 = rectangle, 6 = L shape
    "corner_margin": 2,            # up to 2 corners more: allowed, but a lower score; more than that: the option is dropped
    "smooth_m": 0.45,              # every room: steps and bumps smaller than this go to the neighbouring room
    "corner_tolerance_m": 0.6,     # steps smaller than this (often along the outline of the flat) do not count as corners
    "straighten_m": [0.9, 1.5, 2.1],   # a room with too many corners: arms narrower than this go to the neighbour
    "wall_threshold": 0.35,        # the model's own wall layer: above this value a pixel is a wall between two rooms
    "min_room_m2": {"living": 6.0, "bedroom": 5.0, "bathroom": 1.8, "kitchen": 3.5, "balcony": 1.5},
    "min_rectangle_m2": 1.0,       # the smallest rectangle when the picture is cut into rectangles
    "max_rectangles": 60,
    "fine": 5,                     # each model pixel becomes 5 x 5 small cells (about 2.5 cm) to draw the walls
    "different_options": 0.25,     # two options must differ in at least 25% of the pixels
    "max_kitchens": 1,             # extra kitchens join the room they touch most
    "max_extra_bathrooms": 1,      # at most (bedrooms + 1) bathrooms; the smallest extra ones join a neighbour
}


# ============================================================================================
# Step 2: the model
# ============================================================================================
def condition_vector(bedrooms, balcony, area_m2, area_mean, area_std):
    onehot = np.zeros(BED_MAX, dtype=np.float32)
    onehot[min(max(int(bedrooms), 1), BED_MAX) - 1] = 1
    return np.r_[onehot, float(balcony), (np.log(area_m2) - area_mean) / area_std].astype(np.float32)


if torch is not None:
    def _down(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU())

    def _up(cin, cout):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU())

    class LayoutCVAE(nn.Module):
        # The same model as in 04_layout_generator.ipynb (the saved weights fit it)
        def __init__(self, latent_dim=32, n_cond=BED_MAX + 2, n_out=len(CHANNELS)):
            super().__init__()
            self.latent_dim = latent_dim
            self.enc = nn.Sequential(_down(n_out + 1, 32), _down(32, 64), _down(64, 128), _down(128, 128), nn.Flatten())
            self.to_mu = nn.Linear(128 * 8 * 8 + n_cond, latent_dim)
            self.to_logvar = nn.Linear(128 * 8 * 8 + n_cond, latent_dim)
            self.sil64, self.sil32, self.sil16, self.sil8 = _down(1, 16), _down(16, 32), _down(32, 64), _down(64, 64)
            self.start = nn.Sequential(nn.Linear(latent_dim + n_cond, 128 * 8 * 8), nn.ReLU())
            self.up16 = _up(128 + 64, 128)
            self.up32 = _up(128 + 64, 64)
            self.up64 = _up(64 + 32, 32)
            self.up128 = nn.Upsample(scale_factor=2, mode="nearest")
            self.out = nn.Conv2d(32 + 16 + 1, n_out, 3, padding=1)

        def encode(self, x, s, c):
            h = torch.cat([self.enc(torch.cat([x, s], 1)), c], 1)
            return self.to_mu(h), self.to_logvar(h)

        def decode(self, z, s, c):
            s64 = self.sil64(s)
            s32 = self.sil32(s64)
            s16 = self.sil16(s32)
            s8 = self.sil8(s16)
            h = self.start(torch.cat([z, c], 1)).view(-1, 128, 8, 8)
            h = self.up16(torch.cat([h, s8], 1))
            h = self.up32(torch.cat([h, s16], 1))
            h = self.up64(torch.cat([h, s32], 1))
            h = self.up128(torch.cat([h, s64], 1))
            return torch.sigmoid(self.out(torch.cat([h, s], 1))) * s

        def forward(self, x, s, c):
            mu, logvar = self.encode(x, s, c)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decode(z, s, c), mu, logvar

    class _FiLM(nn.Module):
        # the condition (bedrooms, balcony, area) scales and shifts every feature map: it reaches every layer
        def __init__(self, n_emb, channels):
            super().__init__()
            self.lin = nn.Linear(n_emb, 2 * channels)

        def forward(self, h, e):
            gamma, beta = self.lin(e).chunk(2, dim=1)
            return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    class _UpBlock(nn.Module):
        # double the size, one or two convolutions, then the condition (FiLM)
        def __init__(self, cin, cout, n_emb, convs=2):
            super().__init__()
            layers = [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
            for _ in range(convs - 1):
                layers += [nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU()]
            self.body = nn.Sequential(*layers)
            self.film = _FiLM(n_emb, cout)

        def forward(self, h, e):
            return torch.relu(self.film(self.body(h), e))

    class LayoutCVAE2(nn.Module):
        # v2 (07_layout_generator_v2.ipynb): the same inputs and outputs as LayoutCVAE, but the condition reaches
        # every layer of the decoder (FiLM), and the decoder is wider at 8-32 px. The 128 px step stays light.
        def __init__(self, latent_dim=32, n_cond=BED_MAX + 2, n_out=len(CHANNELS), n_emb=64):
            super().__init__()
            self.latent_dim = latent_dim
            self.enc = nn.Sequential(_down(n_out + 1, 32), _down(32, 64), _down(64, 128), _down(128, 128), nn.Flatten())
            self.to_mu = nn.Linear(128 * 8 * 8 + n_cond, latent_dim)
            self.to_logvar = nn.Linear(128 * 8 * 8 + n_cond, latent_dim)
            self.emb = nn.Sequential(nn.Linear(n_cond, n_emb), nn.ReLU(), nn.Linear(n_emb, n_emb), nn.ReLU())
            self.sil64, self.sil32, self.sil16, self.sil8 = _down(1, 16), _down(16, 32), _down(32, 64), _down(64, 64)
            self.start = nn.Sequential(nn.Linear(latent_dim + n_emb, 192 * 8 * 8), nn.ReLU())
            self.up16 = _UpBlock(192 + 64, 192, n_emb, convs=2)
            self.up32 = _UpBlock(192 + 64, 96, n_emb, convs=2)
            self.up64 = _UpBlock(96 + 32, 48, n_emb, convs=1)
            self.up128 = nn.Upsample(scale_factor=2, mode="nearest")
            self.out = nn.Conv2d(48 + 16 + 1, n_out, 3, padding=1)

        def encode(self, x, s, c):
            h = torch.cat([self.enc(torch.cat([x, s], 1)), c], 1)
            return self.to_mu(h), self.to_logvar(h)

        def decode(self, z, s, c):
            e = self.emb(c)
            s64 = self.sil64(s)
            s32 = self.sil32(s64)
            s16 = self.sil16(s32)
            s8 = self.sil8(s16)
            h = self.start(torch.cat([z, e], 1)).view(-1, 192, 8, 8)
            h = self.up16(torch.cat([h, s8], 1), e)
            h = self.up32(torch.cat([h, s16], 1), e)
            h = self.up64(torch.cat([h, s32], 1), e)
            h = self.up128(torch.cat([h, s64], 1))
            return torch.sigmoid(self.out(torch.cat([h, s], 1))) * s

        def forward(self, x, s, c):
            mu, logvar = self.encode(x, s, c)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decode(z, s, c), mu, logvar

    def make_model(config):
        # the right model class for a saved config: "architecture" v1 (LayoutCVAE) or v2 (LayoutCVAE2)
        cls = LayoutCVAE2 if config.get("architecture", "v1") == "v2" else LayoutCVAE
        return cls(latent_dim=config["latent_dim"])

    @torch.no_grad()
    def generate(model, sil_u8, cond, n=30, seed=None, device="cpu"):
        # sil_u8: (128, 128) values 0-255; cond: (7,)  ->  (n, 8, 128, 128) numpy, values 0-1
        model.eval()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(n, model.latent_dim, generator=gen)
        s = torch.as_tensor(np.asarray(sil_u8), dtype=torch.float32).view(1, 1, IMG, IMG).repeat(n, 1, 1, 1) / 255
        c = torch.as_tensor(np.asarray(cond), dtype=torch.float32).view(1, -1).repeat(n, 1)
        return model.decode(z.to(device), s.to(device), c.to(device)).cpu().numpy()


def to_color(maps, sil=None, room_threshold=0.5, line_threshold=0.35):
    # (8, H, W) room map -> RGB picture (for figures)
    from matplotlib.colors import to_rgb
    maps = np.asarray(maps, dtype=np.float32)
    if maps.max() > 1.5:
        maps = maps / 255
    rgb = np.ones((maps.shape[1], maps.shape[2], 3))
    if sil is not None:
        rgb[np.asarray(sil) > 127] = to_rgb("#e4e3dc")
    rooms = maps[:len(ROOMS)]
    label, strong = rooms.argmax(0), rooms.max(0) > room_threshold
    for k, name in enumerate(ROOMS):
        rgb[(label == k) & strong] = to_rgb(COLORS[name])
    for name in ["wall", "window", "door"]:
        rgb[maps[CHANNELS.index(name)] > line_threshold] = to_rgb(COLORS[name])
    return rgb


# ============================================================================================
# Step 3: from the model's picture to a clean plan
# ============================================================================================
def room_labels(maps, inside, strong=0.3):
    # 1. Every pixel inside the silhouette gets one room type. Weak pixels (grey, blurry) take the type of the nearest strong pixel.
    rooms = np.asarray(maps[:len(ROOMS)], dtype=np.float32)
    label = rooms.argmax(0)
    ok = (rooms.max(0) >= strong) & inside
    if ok.any() and (inside & ~ok).any():
        _, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
        label = np.where(ok, label, label[iy, ix])
    return np.where(inside, label, -1)


def rectangles(label, m_per_px, settings=SETTINGS):
    # 2. Cut the picture into rectangles with straight horizontal and vertical lines (a decision tree on x, y).
    #    Each rectangle gets one room type, or "outside" (-1): the walls between rooms AND the outline become straight
    #    (small steps in the outline, smaller than min_rectangle_m2, disappear).
    ys, xs = np.nonzero(label >= 0)
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, label.shape[0])
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, label.shape[1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    X = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    min_px = max(4, int(settings["min_rectangle_m2"] / m_per_px ** 2))
    tree = DecisionTreeClassifier(max_leaf_nodes=settings["max_rectangles"], min_samples_leaf=min_px, random_state=0)
    tree.fit(X, label[y0:y1, x0:x1].ravel())
    out = np.full(label.shape, -1)
    out[y0:y1, x0:x1] = tree.predict(X).reshape(y1 - y0, x1 - x0)
    return out


def _components(rect_label):
    # one id per room: connected pieces of the same type
    inst = np.full(rect_label.shape, -1)
    types = []
    for t in range(len(ROOMS)):
        n, lab = cv2.connectedComponents((rect_label == t).astype(np.uint8), connectivity=4)
        for k in range(1, n):
            inst[lab == k] = len(types)
            types.append(ROOMS[t])
    return inst, types


def _renumber(inst, types):
    ids = [i for i in range(len(types)) if (inst == i).any()]
    new = np.full(inst.shape, -1)
    for j, i in enumerate(ids):
        new[inst == i] = j
    return new, [types[i] for i in ids]


def _merge_into_neighbour(inst, i, inside):
    # give room i to the neighbouring room it touches most
    mask = inst == i
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~mask & inside
    neighbours = inst[ring]
    neighbours = neighbours[neighbours >= 0]
    if len(neighbours):
        inst[mask] = Counter(neighbours.tolist()).most_common(1)[0][0]
    return inst


def clean_rooms(rect_label, inside, m_per_px, settings=SETTINGS):
    # 3. Rooms = connected pieces. Too small pieces join their neighbour; a "balcony" that does not touch the outside joins its neighbour.
    inst, types = _components(rect_label)
    px_m2 = m_per_px ** 2
    outside = ~inside
    for _ in range(3):
        changed = False
        for i, t in enumerate(types):
            mask = inst == i
            if not mask.any():
                continue
            too_small = mask.sum() * px_m2 < settings["min_room_m2"].get(t, 1.0)
            touches_outside = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & outside).sum()
            inner_balcony = t == "balcony" and touches_outside * m_per_px < 1.0
            if too_small or inner_balcony:
                inst = _merge_into_neighbour(inst, i, inside)
                changed = True
        inst, types = _renumber(inst, types)
        if not changed:
            break
    return inst, types


def _split(mask):
    # cut a room in two equal halves, across its long side
    ys, xs = np.nonzero(mask)
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    coord = xs if horizontal else ys
    cut = np.sort(coord)[len(coord) // 2]
    grid = np.arange(mask.shape[1])[None, :] if horizontal else np.arange(mask.shape[0])[:, None]
    a = mask & (grid < cut)
    return a, mask & ~a


def _size_m(mask, m_per_px):
    ys, xs = np.nonzero(mask)
    return mask.sum() * m_per_px ** 2, min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * m_per_px


def _net_size(mask, envelope, m_per_px, settings=SETTINGS):
    # area and width of a room INSIDE its walls, estimated before the walls are drawn:
    # an exterior wall (towards the outside or a balcony) takes its full 20 cm from the room, an interior wall half of its 10 cm
    area, width = _size_m(mask, m_per_px)
    m, env = np.pad(mask.astype(bool), 1), np.pad(envelope.astype(bool), 1)
    h, w = m.shape
    lost = 0.0
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nb_room = m[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        nb_env = env[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        edge = m[1:-1, 1:-1] & ~nb_room
        lost += (edge & ~nb_env).sum() * settings["exterior_wall_m"] + (edge & nb_env).sum() * settings["interior_wall_m"] / 2
    return area - lost * m_per_px, width - settings["interior_wall_m"]


def corners(mask):
    # number of corners of a shape on the pixel grid (outline and holes): 4 = rectangle, 6 = L shape
    m = np.pad(np.asarray(mask, dtype=bool), 1).astype(np.int8)
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    total = a + b + c + d
    diagonal = (a == d) & (b == c) & (a != b)
    return int(((total == 1) | (total == 3)).sum() + 2 * diagonal.sum())


def visible_corners(mask, m_per_px, settings=SETTINGS):
    # corners that a person sees: small steps (smaller than corner_tolerance_m) are not counted
    m = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    if not m.any():
        return 0
    up = cv2.resize(m, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)       # contours on pixel edges, not centres
    contours, _ = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    eps = 2 * 0.75 * settings["corner_tolerance_m"] / m_per_px                  # in the doubled grid
    return int(sum(len(cv2.approxPolyDP(c, eps, True)) for c in contours if len(c) >= 4))


def straighten_rooms(inst, types, m_per_px, settings=SETTINGS, skip=("living",)):
    # 4d. First every room loses its small steps and bumps (smaller than 45 cm). Then a room with too many corners
    #     loses its thin arms (narrower than 0.9 m, then 1.5 m, then 2.1 m). The pieces go to the room they touch most;
    #     the living room is not cut in the second pass: it takes the pieces.
    inst = inst.copy()
    inside = inst >= 0
    passes = [(i, [settings["smooth_m"]], True) for i in range(len(types))] + \
             [(i, settings["straighten_m"], False) for i, t in enumerate(types) if t not in skip]
    for i, widths, always in passes:
        t = types[i]
        for width_m in widths:
            mask = inst == i
            if not mask.any() or (not always and visible_corners(mask, m_per_px, settings) <= settings["max_corners"].get(t, 8)):
                break
            k = max(3, int(round(width_m / m_per_px)) | 1)
            core = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
            n, lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
            if n < 2:
                break
            core = lab == (np.argmax(np.bincount(lab[lab > 0])) if n > 2 else 1)
            if core.sum() < 0.6 * mask.sum():                # it would lose too much: leave it
                break
            n, lab = cv2.connectedComponents((mask & ~core).astype(np.uint8), connectivity=4)
            for piece_id in range(1, n):
                piece = lab == piece_id
                ring = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~mask & inside
                neighbours = inst[ring]
                neighbours = neighbours[neighbours >= 0]
                if len(neighbours):
                    inst[piece] = Counter(neighbours.tolist()).most_common(1)[0][0]
    return _renumber(inst, types)


def _split_by_walls(mask, walls, settings=SETTINGS):
    # the model drew a wall across this room: the two biggest pieces on each side of it, or None
    if walls is None:
        return None
    open_ = mask & (walls < settings["wall_threshold"])
    n, lab = cv2.connectedComponents(open_.astype(np.uint8), connectivity=4)
    if n < 3:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    first, second = np.argsort(-sizes)[:2] + 1
    seeds = (lab == first) | (lab == second)
    _, (iy, ix) = ndimage.distance_transform_edt(~seeds, return_indices=True)
    nearest = lab[iy, ix]                                      # every other pixel of the room joins the nearest piece
    return mask & (nearest == first), mask & (nearest == second)


def fix_bedrooms(inst, types, wanted, m_per_px, settings=SETTINGS, walls=None):
    # 4. Exactly the asked number of bedrooms: split the largest bedroom in two (equal halves) until the count is right.
    #    Returns None and a reason when it is not possible.
    inst, types = inst.copy(), list(types)
    envelope = (inst >= 0) & ~np.isin(inst, [i for i, t in enumerate(types) if t == "balcony"])
    while True:
        beds = [i for i, t in enumerate(types) if t == "bedroom"]
        if len(beds) > wanted:
            return None, None, "too many bedrooms"
        if len(beds) == wanted:
            break
        done = False
        for i in sorted(beds, key=lambda i: -(inst == i).sum()):
            split = _split_by_walls(inst == i, walls, settings)        # first: along a wall the model drew
            if split is not None:
                a, b = split
                (area_a, width_a), (area_b, width_b) = _net_size(a, envelope, m_per_px, settings), _net_size(b, envelope, m_per_px, settings)
                if min(area_a, area_b) >= settings["min_bedroom_m2"] and min(width_a, width_b) >= settings["min_bedroom_width_m"]:
                    inst[b] = len(types)
                    types.append("bedroom")
                    done = True
                    break
            a, b = _split(inst == i)                                   # else: in two equal halves
            if not a.any() or not b.any():
                continue
            if cv2.connectedComponents(a.astype(np.uint8), connectivity=4)[0] > 2 or \
               cv2.connectedComponents(b.astype(np.uint8), connectivity=4)[0] > 2:
                continue
            (area_a, width_a), (area_b, width_b) = _net_size(a, envelope, m_per_px, settings), _net_size(b, envelope, m_per_px, settings)
            if min(area_a, area_b) < settings["min_bedroom_m2"] or min(width_a, width_b) < settings["min_bedroom_width_m"]:
                continue
            inst[b] = len(types)
            types.append("bedroom")
            done = True
            break
        if not done:
            return None, None, f"cannot fit {wanted} bedrooms"
    for i, t in enumerate(types):              # every bedroom big enough (net, inside the walls)
        if t == "bedroom":
            area, width = _net_size(inst == i, envelope, m_per_px, settings)
            if area < settings["min_bedroom_m2"] or width < settings["min_bedroom_width_m"]:
                return None, None, "a bedroom is too small"
    return inst, types, ""


def split_big_rooms(inst, types, m_per_px, rules, settings=SETTINGS, kinds=("bathroom",)):
    # 4b. A bathroom much bigger than usual (more than 1.5 x the "large (90%)" size in the room table) is
    #     usually two bathrooms that merged: split it in two equal halves.
    sizes = rules.get("room_m2", {})
    for kind in kinds:
        if kind not in sizes:
            continue
        limit = 1.5 * sizes[kind]["large (90%) m²"]
        changed = True
        while changed:
            changed = False
            for i, t in enumerate(types):
                mask = inst == i
                if t != kind or mask.sum() * m_per_px ** 2 <= limit:
                    continue
                a, b = _split(mask)
                if a.any() and b.any() and min(a.sum(), b.sum()) * m_per_px ** 2 >= settings["min_room_m2"].get(kind, 1.0):
                    inst[b] = len(types)
                    types.append(kind)
                    changed = True
                    break
    return inst, types


def limit_rooms(inst, types, bedrooms, settings=SETTINGS):
    # 4c. One kitchen, and at most (bedrooms + 1) bathrooms: the smallest extra ones join the room they touch most
    inside = inst >= 0
    limits = {"kitchen": settings["max_kitchens"], "bathroom": int(bedrooms) + settings["max_extra_bathrooms"]}
    for kind, limit in limits.items():
        while True:
            ids = [i for i, t in enumerate(types) if t == kind and (inst == i).any()]
            if len(ids) <= limit:
                break
            smallest = min(ids, key=lambda i: (inst == i).sum())
            before = (inst == smallest).sum()
            inst = _merge_into_neighbour(inst, smallest, inside)
            if (inst == smallest).sum() == before:          # no neighbour to join: stop
                break
    return _renumber(inst, types)


def _runs(inst, a, b):
    # straight pieces of the border between room a and room b (b = -1: the outside).
    # Returns a list of (orientation, line, start, end) in pixels; "h" = a horizontal border at y = line.
    runs = []
    for orient in ["h", "v"]:
        if orient == "h":
            p, q = inst[:-1, :], inst[1:, :]
        else:
            p, q = inst[:, :-1].T, inst[:, 1:].T
        hit = ((p == a) & (q == b)) | ((p == b) & (q == a))
        for line in np.nonzero(hit.any(1))[0]:
            cols = np.nonzero(hit[line])[0]
            breaks = np.nonzero(np.diff(cols) > 1)[0]
            starts = np.r_[cols[0], cols[breaks + 1]]
            ends = np.r_[cols[breaks], cols[-1]] + 1
            runs += [(orient, line + 1, s, e) for s, e in zip(starts, ends)]
    return sorted(runs, key=lambda r: -(r[3] - r[2]))


def _opening(run, width_m, cell, fine, depth_cells, walls, at=0.5):
    # a rectangle across the wall on a border run: the cells of the wall it cuts
    orient, line, start, end = run
    length = (end - start) * fine
    w = int(round(width_m / cell))
    centre = start * fine + at * length
    a0, a1 = int(round(centre - w / 2)), int(round(centre + w / 2))
    l0, l1 = line * fine - depth_cells, line * fine + depth_cells
    rect = np.zeros_like(walls)
    if orient == "h":
        rect[max(l0, 0):l1, max(a0, 0):a1] = True
    else:
        rect[max(a0, 0):a1, max(l0, 0):l1] = True
    return rect & walls


def walls_doors_windows(inst, types, m_per_px, rules, settings=SETTINGS):
    # 5. Walls (exterior 20 cm, interior 10 cm), doors chosen with the room rules, windows on the outer walls.
    fine = settings["fine"]
    cell = m_per_px / fine
    inst_f = np.repeat(np.repeat(inst, fine, axis=0), fine, axis=1)
    inside_f = inst_f >= 0
    balcony_ids = [i for i, t in enumerate(types) if t == "balcony"]
    balcony_f = np.isin(inst_f, balcony_ids)
    envelope = inside_f & ~balcony_f
    ext = max(1, int(round(settings["exterior_wall_m"] / cell)))
    half = max(1, int(round(settings["interior_wall_m"] / 2 / cell)))

    padded = np.pad(envelope.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]
    walls = envelope & (dist <= ext)                                   # exterior walls (also towards the balconies)
    kernel = np.ones((2 * half + 1, 2 * half + 1), np.uint8)
    core = np.zeros_like(envelope)
    for i, t in enumerate(types):
        if t != "balcony":
            core |= cv2.erode((inst_f == i).astype(np.uint8), kernel).astype(bool)
    walls |= envelope & ~core                                          # interior walls: 10 cm on the border between rooms
    solid = walls.copy()                                               # the walls before the doors and windows are cut
    bal_pad = np.pad(inside_f.astype(np.uint8), 1)
    rails = balcony_f & (cv2.distanceTransform(bal_pad, cv2.DIST_C, 3)[1:-1, 1:-1] <= 2)

    # --- doors: start from the living room, and each time open the most typical door to a room not reached yet ---
    opens = rules.get("opens_to_percent", {})
    living = [i for i, t in enumerate(types) if t == "living"]
    root = max(living or range(len(types)), key=lambda i: (inst == i).sum())
    reached, doors, door_info = {root}, [], []
    border = {}
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            r = _runs(inst, i, j)
            if r:
                border[(i, j)] = border[(j, i)] = r[0]
    while len(reached) < len(types):
        best = None
        for v in range(len(types)):
            if v in reached:
                continue
            for u in reached:
                run = border.get((u, v))
                if run is None:
                    continue
                width = settings["door_m"].get(types[v], 0.8)
                if (run[3] - run[2]) * m_per_px < width + 0.2:
                    continue
                score = opens.get(types[v], {}).get(types[u], 0.0) + 0.01 * (run[3] - run[2])
                if best is None or score > best[0]:
                    best = (score, u, v, run, width)
        if best is None:
            break
        score, u, v, run, width = best
        depth = ext + 1 if "balcony" in (types[u], types[v]) else half + 1
        cut = _opening(run, width, cell, fine, depth, walls)
        walls &= ~cut
        doors.append(cut)
        door_info.append({"from": types[u], "to": types[v], "width_m": width,
                          "typical_percent": opens.get(types[v], {}).get(types[u], 0.0)})
        reached.add(v)
    unreached = [types[i] for i in range(len(types)) if i not in reached]

    # --- windows: on the longest outer border of each room; the entrance on another outer border of the living room ---
    windows, windows_of, entrance = [], set(), None
    for i, t in enumerate(types):
        if t not in settings["window_m"]:
            continue
        runs = [r for r in _runs(inst, i, -1)]
        if not runs:
            continue
        run = runs[0]
        length_m = (run[3] - run[2]) * m_per_px
        width = min(settings["window_m"][t], 0.6 * length_m)
        if width >= 0.6:
            cut = _opening(run, width, cell, fine, ext + 1, walls)
            walls &= ~cut
            windows.append(cut)
            windows_of.add(i)
        if i == root and len(runs) > 1 and (runs[1][3] - runs[1][2]) * m_per_px >= settings["entrance_m"] + 0.3:
            entrance = _opening(runs[1], settings["entrance_m"], cell, fine, ext + 1, walls)
            walls &= ~entrance

    rooms_f = [(t, (inst_f == i) & ~solid) for i, t in enumerate(types)]   # net rooms: up to the wall face, openings not counted
    return {"cell": cell, "walls": walls, "rooms": rooms_f, "doors": doors, "windows": windows, "rails": rails,
            "entrance": entrance, "outline": inside_f, "door_info": door_info, "unreached": unreached,
            "windows_of": windows_of}


def score_layout(types, parts, m_per_px, rules, inst=None, settings=SETTINGS):
    # 6. A score from 0 to 1: typical room sizes, typical doors, windows in bedrooms and living room, rectangular rooms,
    #    simple shapes (no more corners than max_corners)
    cell = parts["cell"]
    sizes = rules.get("room_m2", {})
    size_s, shape_s = [], []
    for t, mask in parts["rooms"]:
        if not mask.any():
            continue
        area = mask.sum() * cell ** 2
        s = sizes.get(t)
        if s:
            lo, hi = s["small (10%) m²"], s["large (90%) m²"]
            size_s.append(1.0 if lo <= area <= hi else (area / lo if area < lo else hi / area))
        if t != "balcony":
            ys, xs = np.nonzero(mask)
            shape_s.append(mask.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))
    door_s = [d["typical_percent"] / 100 for d in parts["door_info"] if d["from"] != d["to"]]
    need = [i for i, t in enumerate(types) if t in ("bedroom", "living")]
    window_s = len([i for i in need if i in parts["windows_of"]]) / max(len(need), 1)
    parts_score = {"sizes": float(np.mean(size_s)) if size_s else 0.0,
                   "doors": float(np.mean(door_s)) if door_s else 0.0,
                   "windows": float(window_s),
                   "rectangular": float(np.mean(shape_s)) if shape_s else 0.0}
    if inst is not None:
        simple = [min(1.0, settings["max_corners"].get(t, 8) / max(visible_corners(inst == i, m_per_px, settings), 1))
                  for i, t in enumerate(types)]
        parts_score["simple shapes"] = float(np.mean(simple)) if simple else 0.0
    return float(np.mean(list(parts_score.values()))), parts_score


def _offset(points, sign):
    # move every corner of a straight-edged outline half a cell outwards (sign +1 or -1)
    p = points.astype(float)
    d_in = p - np.roll(p, 1, 0)
    d_out = np.roll(p, -1, 0) - p
    unit = lambda d: d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    return p + sign * 0.5 * (unit(d_in) @ rot + unit(d_out) @ rot)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _straighten(pts, mask):
    # the outline found by OpenCV goes through cell centres and takes diagonal steps at inner corners:
    # replace each diagonal step by the corner cell that belongs to the shape, then keep only the corners
    out = [tuple(pts[0])]
    for q in [tuple(p) for p in pts[1:]] + [tuple(pts[0])]:
        p = out[-1]
        if p[0] != q[0] and p[1] != q[1]:
            a, b = (q[0], p[1]), (p[0], q[1])
            out.append(a if mask[int(a[1]), int(a[0])] else b)
        if q != out[-1]:
            out.append(q)
    out = out[:-1] if len(out) > 1 and out[-1] == out[0] else out
    n = len(out)
    keep = [out[i] for i in range(n)
            if not ((out[i - 1][0] == out[i][0] == out[(i + 1) % n][0]) or
                    (out[i - 1][1] == out[i][1] == out[(i + 1) % n][1]))]
    return np.array(keep if len(keep) >= 4 else out, dtype=float)


def _open2(mask):
    # keep only the cells covered by a full 2 x 2 block of the shape (OpenCV's own opening shifts even kernels)
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(cv2.erode(mask, k, anchor=(0, 0), borderValue=0), k, anchor=(1, 1))


def mask_to_polygons(mask, cell, min_cells=4):
    # 7. Small cells -> polygons in metres (y up). Each polygon = [outer ring, hole rings...], ring = (n, 2) array.
    mask = np.asarray(mask, dtype=np.uint8)
    mask = _open2(mask)                                                          # drop slivers one cell wide
    if mask.sum() < min_cells:
        return []
    height = mask.shape[0]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    rings = []
    for k, c in enumerate(contours):
        pts = c[:, 0, :]
        if len(pts) < 3 or len(set(pts[:, 0])) == 1 or len(set(pts[:, 1])) == 1:     # a line of cells: use its box
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        else:
            pts = _straighten(pts, mask)
        is_hole = hierarchy[k][3] >= 0
        a = _offset(pts, +1)                       # the outline goes through cell centres: move it to the cell edges
        b = _offset(pts, -1)
        grow = a if _area(a) > _area(b) else b
        shrink = b if grow is a else a
        ring = shrink if is_hole else grow
        ring = np.c_[(ring[:, 0] + 0.5) * cell, (height - ring[:, 1] - 0.5) * cell]
        rings.append((k, is_hole, hierarchy[k][3], ring))
    polygons = []
    for k, is_hole, parent, ring in rings:
        if not is_hole:
            polygons.append([ring] + [r for kk, h, par, r in rings if h and par == k])
    return polygons


def _plan_rooms(maps, sil_u8, bedrooms, balcony, m_per_px, rules, settings=SETTINGS):
    # steps 1-4c: the model's picture -> rooms on the model's grid (inst, types), or a reason why it was dropped
    label = room_labels(maps, np.asarray(sil_u8) > 127)
    rect = rectangles(label, m_per_px, settings)
    inst, types = clean_rooms(rect, rect >= 0, m_per_px, settings)
    debug = {"label": label, "rect": rect, "inst_before_split": inst.copy(), "types_before_split": list(types)}
    if bool(balcony) != ("balcony" in types):
        return None, None, "balcony asked" if balcony else "balcony not asked", debug
    walls = np.asarray(maps[CHANNELS.index("wall")], dtype=np.float32)
    inst, types, why = fix_bedrooms(inst, types, int(bedrooms), m_per_px, settings, walls=walls)
    if inst is None:
        return None, None, why, debug
    inst, types = split_big_rooms(inst, types, m_per_px, rules, settings)
    inst, types = limit_rooms(inst, types, bedrooms, settings)
    inst, types = straighten_rooms(inst, types, m_per_px, settings)
    missing = [t for t in ["living", "kitchen", "bathroom"] if t not in types]
    if missing:                                          # every flat needs a living room, a kitchen and a bathroom
        return None, None, "no " + missing[0], debug
    for i, t in enumerate(types):
        if visible_corners(inst == i, m_per_px, settings) > settings["max_corners"].get(t, 8) + settings["corner_margin"]:
            return None, None, f"a {t} has too many corners", debug
    outside = inst < 0
    for i, t in enumerate(types):                        # the kitchen is on an outer wall (window, ventilation)
        if t == "kitchen":
            k = (inst == i).astype(np.uint8)
            contact = sum(int((np.roll(k, shift, axis) & outside).sum()) for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)])
            if contact * m_per_px < settings["kitchen_outer_wall_m"]:
                return None, None, "kitchen not on an outer wall", debug
    return inst, types, "", debug


def make_layout(maps, sil_u8, bedrooms, balcony, m_per_px, rules, settings=SETTINGS, vector=True, debug=False):
    # The whole of step 3 for one picture of the model. Returns a layout (vectors in metres) or a reason why it was dropped.
    layout = {"ok": False, "reason": "", "bedrooms_asked": int(bedrooms), "balcony_asked": int(balcony)}
    k = ROOMS.index("kitchen")
    for boost in settings["kitchen_boost"]:
        # The model often draws the kitchen weakly next to the living room, and the living room wins (or a small
        # kitchen floats inside it). Then the kitchen signal of the SAME picture is made stronger, and steps 1-4d run again.
        m = np.array(maps, dtype=np.float32)
        if boost != 1.0:
            m[k] = np.clip(m[k] * boost, 0, 1)
        inst, types, why, dbg = _plan_rooms(m, sil_u8, bedrooms, balcony, m_per_px, rules, settings)
        if why not in ("no kitchen", "kitchen not on an outer wall"):
            break
    if debug:
        layout["debug"] = dbg
    if inst is None:
        layout["reason"] = why
        return layout
    layout["kitchen_boost"] = boost
    parts = walls_doors_windows(inst, types, m_per_px, rules, settings)
    if parts["unreached"]:
        layout["reason"] = "no door to: " + ", ".join(parts["unreached"])
        return layout
    cell = parts["cell"]
    for t, mask in parts["rooms"]:                       # the real sizes, inside the walls
        if t == "bedroom":
            ys, xs = np.nonzero(mask)
            width = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * cell if len(ys) else 0.0
            if mask.sum() * cell ** 2 < settings["min_bedroom_m2"] or width < settings["min_bedroom_width_m"]:
                layout["reason"] = "a bedroom is too small"
                return layout
    bathrooms = [mask.sum() * cell ** 2 for t, mask in parts["rooms"] if t == "bathroom"]
    if max(bathrooms, default=0.0) < settings["min_main_bathroom_m2"]:
        layout["reason"] = "main bathroom too small"
        return layout
    score, score_parts = score_layout(types, parts, m_per_px, rules, inst=inst, settings=settings)
    layout.update({"ok": True, "score": score, "score_parts": score_parts, "room_types": types, "inst": inst,
                   "doors_info": parts["door_info"]})
    if debug:
        layout["debug"].update({"inst": inst, "types": types, "parts": parts})
    if not vector:
        return layout
    layout.update({
        "rooms": [{"type": t, "area_m2": round(float(m.sum() * cell ** 2), 1), "polygons": mask_to_polygons(m, cell)}
                  for t, m in parts["rooms"]],
        "walls": mask_to_polygons(parts["walls"], cell),
        "doors": [p for d in parts["doors"] for p in mask_to_polygons(d, cell)],
        "windows": [p for w in parts["windows"] for p in mask_to_polygons(w, cell)],
        "rails": mask_to_polygons(parts["rails"], cell),
        "entrance": mask_to_polygons(parts["entrance"], cell) if parts["entrance"] is not None else [],
        "outline": mask_to_polygons(parts["outline"], cell),
    })
    return layout


def design_from_maps(maps, sil_u8, bedrooms, balcony, m_per_px, rules, n_options=5, settings=SETTINGS):
    # Step 3 for one request: maps = the model's pictures (n, 8, 128, 128). Returns the options and why the others were dropped.
    light = [make_layout(m, sil_u8, bedrooms, balcony, m_per_px, rules, settings, vector=False) for m in maps]
    for k, l in enumerate(light):
        l["sample"] = k
    chosen = choose_options(light, n_options, settings)
    options = []
    for l in chosen:                                       # only the chosen ones become vectors
        full = make_layout(maps[l["sample"]], sil_u8, bedrooms, balcony, m_per_px, rules, settings, vector=True)
        full["sample"] = l["sample"]
        options.append(full)
    reasons = Counter("ok" if l["ok"] else l["reason"].split(" (")[0].split(":")[0] for l in light)
    return options, reasons


def choose_options(layouts, n_options=5, settings=SETTINGS):
    # 8. The best layouts that are also different from each other
    good = sorted([l for l in layouts if l["ok"]], key=lambda l: -l["score"])
    chosen = []
    for l in good:
        type_index = np.array([ROOMS.index(t) for t in l["room_types"]])
        lab = np.where(l["inst"] >= 0, type_index[np.maximum(l["inst"], 0)], -1).ravel()
        inside = lab >= 0
        if all((lab[inside] != c[1][inside]).mean() >= settings["different_options"] for c in chosen):
            chosen.append((l, lab))
        if len(chosen) == n_options:
            break
    for l in good:                                # not enough different ones: fill with the next best
        if len(chosen) == n_options:
            break
        if all(l is not c[0] for c in chosen):
            chosen.append((l, None))
    return [c[0] for c in chosen]


# ============================================================================================
# Drawing and DXF
# ============================================================================================
def _patch(polygon, **kw):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    verts, codes = [], []
    for ring in polygon:
        ring = np.asarray(ring)
        verts += ring.tolist() + [ring[0].tolist()]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1) + [Path.CLOSEPOLY]
    return PathPatch(Path(verts, codes), **kw)


def label_point(polygon, px_per_m=20):
    # the point of a room farthest from its edges (never in a hole, never outside an L shape)
    rings = [np.asarray(r, dtype=float) for r in polygon]
    low, high = rings[0].min(0), rings[0].max(0)
    w, h = np.ceil((high - low) * px_per_m).astype(int) + 3
    img = np.zeros((h, w), np.uint8)
    to_px = lambda r: np.round((r - low) * px_per_m + 1).astype(np.int32)
    cv2.fillPoly(img, [to_px(rings[0])], 1)
    for hole in rings[1:]:
        cv2.fillPoly(img, [to_px(hole)], 0)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (x - 1) / px_per_m + low[0], (y - 1) / px_per_m + low[1]


def draw_layout(layout, ax, title="", labels=True):
    for room in layout["rooms"]:
        for poly in room["polygons"]:
            ax.add_patch(_patch(poly, facecolor=COLORS[room["type"]], edgecolor="none", alpha=0.85))
    for poly in layout["walls"]:
        ax.add_patch(_patch(poly, facecolor=COLORS["wall"], edgecolor="none"))
    for poly in layout["windows"]:
        ax.add_patch(_patch(poly, facecolor=COLORS["window"], edgecolor="#3d3d3a", linewidth=0.3))
    for poly in layout["doors"]:
        ax.add_patch(_patch(poly, facecolor="#ffffff", edgecolor="#3d3d3a", linewidth=0.3))
    for poly in layout["entrance"]:
        ax.add_patch(_patch(poly, facecolor=COLORS["entrance"], edgecolor="none"))
    for poly in layout["rails"]:
        ax.add_patch(_patch(poly, facecolor="#9b5f7a", edgecolor="none"))
    if labels:
        for room in layout["rooms"]:
            if room["polygons"] and room["area_m2"] >= 2.5:
                x, y = label_point(max(room["polygons"], key=lambda p: _area(np.asarray(p[0]))))
                ax.text(x, y, f"{room['area_m2']:.0f}", ha="center", va="center", fontsize=6, color="#0b0b0b")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


DXF_LAYERS = {   # name: (AutoCAD colour, line weight in 1/100 mm)
    "A-WALL": (7, 35), "A-WALL-PATT": (8, 0), "A-DOOR": (3, 25), "A-GLAZ": (4, 25), "A-RAIL": (6, 18),
    "A-DOOR-ENTR": (1, 35), "A-AREA-BNDY": (9, 13), "A-AREA-IDEN": (7, 18),
    "A-AREA-LIVING": (5, 18), "A-AREA-BEDROOM": (30, 18), "A-AREA-BATH": (130, 18),
    "A-AREA-KITCHEN": (40, 18), "A-AREA-BALCONY": (210, 18),
}
ROOM_LAYER = {"living": "A-AREA-LIVING", "bedroom": "A-AREA-BEDROOM", "bathroom": "A-AREA-BATH",
              "kitchen": "A-AREA-KITCHEN", "balcony": "A-AREA-BALCONY"}


def export_dxf(layout, path, note=""):
    # One option -> DXF in metres, one layer per element, thin line weights
    import ezdxf
    from ezdxf import units as dxf_units
    doc = ezdxf.new("R2010")
    doc.units = dxf_units.M
    for name, (color, weight) in DXF_LAYERS.items():
        doc.layers.new(name, dxfattribs={"color": color, "lineweight": weight})
    msp = doc.modelspace()

    def rings(poly):
        return [[(round(float(x), 4), round(float(y), 4)) for x, y in ring] for ring in poly]

    def outline(polys, layer, fill=False):
        for poly in polys:
            for ring in rings(poly):
                msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": layer})
            if fill:
                hatch = msp.add_hatch(color=8, dxfattribs={"layer": "A-WALL-PATT"})
                for ring in rings(poly):
                    hatch.paths.add_polyline_path(ring, is_closed=True)

    outline(layout["walls"], "A-WALL", fill=True)
    outline(layout["doors"], "A-DOOR")
    outline(layout["windows"], "A-GLAZ")
    outline(layout["rails"], "A-RAIL")
    outline(layout["entrance"], "A-DOOR-ENTR")
    outline(layout["outline"], "A-AREA-BNDY")
    for room in layout["rooms"]:
        outline(room["polygons"], ROOM_LAYER[room["type"]])
        if room["polygons"] and room["area_m2"] >= 1.0:
            x, y = label_point(max(room["polygons"], key=lambda p: _area(np.asarray(p[0]))))
            msp.add_mtext(f"{room['type'].upper()}\\P{room['area_m2']:.1f} m²",
                          dxfattribs={"layer": "A-AREA-IDEN", "char_height": 0.2,
                                      "insert": (float(x), float(y)), "attachment_point": 5})
    if note:
        msp.add_mtext(note, dxfattribs={"layer": "A-AREA-IDEN", "char_height": 0.3, "insert": (0.0, -0.8),
                                        "attachment_point": 1})
    doc.saveas(str(path))
    return path


# ============================================================================================
# Step 4: from a photo of the silhouette to the model's input
# ============================================================================================
PHOTO_SIZE = 1000          # the photo is resized so its long side is 1000 pixels


def read_photo(source):
    # a file path or the bytes of an uploaded file -> grey image, long side PHOTO_SIZE
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cannot read the photo")
    k = PHOTO_SIZE / max(img.shape)
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)


def _thin(mask):
    # Zhang-Suen thinning: every line becomes one pixel wide (to find the ends of the lines)
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in range(2):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9, c = img[1:-1, :-2], img[:-2, :-2], img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(n.astype(np.int16) for n in ring[:8])
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                a, b = p2 * p4 * p6, p4 * p6 * p8
            else:
                a, b = p2 * p4 * p8, p2 * p6 * p8
            remove = (c == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & (a == 0) & (b == 0)
            if remove.any():
                c[remove] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _ink(gray):
    # dark pen lines -> mask. Dividing by the blurred paper removes shadows; faint grid lines stay below the threshold.
    paper = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    flat = np.clip(gray.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)
    darkness = 255 - flat
    marks = darkness[darkness > 20]                                # pen + grid (not the empty paper)
    level = cv2.threshold(marks.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if marks.size > 100 else 60
    ink = darkness > max(level, 60)                                 # the pen is much darker than the grid
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= 0.04 * max(gray.shape)
    return keep[lab], flat


def _along_line(skel, ends, limit):
    # for each line end: the other ends it reaches by walking ALONG the line (within `limit` pixels), and how far
    H, W = skel.shape
    index = {tuple(e): k for k, e in enumerate(ends)}
    near = {}
    for k, (y, x) in enumerate(ends):
        seen, frontier, steps = {(y, x)}, [(y, x)], 0
        while frontier and steps < limit:
            steps += 1
            nxt = []
            for cy, cx in frontier:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (ny, nx) in seen or not (0 <= ny < H and 0 <= nx < W) or not skel[ny, nx]:
                            continue
                        seen.add((ny, nx))
                        nxt.append((ny, nx))
                        j = index.get((ny, nx))
                        if j is not None and j != k:
                            near[(min(k, j), max(k, j))] = min(steps, near.get((min(k, j), max(k, j)), steps))
            frontier = nxt
    return near


def _close_gaps(ink, max_gap):
    # Join line ends that are close in space but far apart ALONG the line: these are the gaps of the drawing.
    # Ends that are also close along the line (a hook, a small branch at a corner) are not joined.
    skel = _thin(ink)
    count = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    ends = np.argwhere(skel & (count == 2))                       # a pixel with exactly one neighbour
    if len(ends) > 400:                                           # far too many line ends: not a clean drawing
        return ink.astype(bool), []
    from scipy.spatial import cKDTree
    along = _along_line(skel, ends, int(3 * max_gap))
    pairs = []
    for i, j in cKDTree(ends).query_pairs(max_gap):
        d = np.hypot(*(ends[i] - ends[j]))
        if along.get((i, j), np.inf) < 3 * d + 10:                # the same piece of line, close along it: not a gap
            continue
        pairs.append((d, i, j))
    closed, used, links = ink.copy().astype(np.uint8), set(), []
    for d, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        used |= {i, j}
        (y0, x0), (y1, x1) = ends[i], ends[j]
        cv2.line(closed, (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=5)
        links.append(((x0, y0), (x1, y1)))
    return closed.astype(bool), links


def _enclosed(lines):
    # the area that the lines close: fill from the border of the image, what is not reached is inside
    h, w = lines.shape
    flood = np.pad(lines.astype(np.uint8), 1) * 255
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    inside = (flood[1:-1, 1:-1] == 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = lab == best
    if region.sum() < 0.02 * h * w:
        return None
    return cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) | (region & lines)


def square_up(points):
    # Make every edge exactly horizontal or vertical. points: (n, 2) corners in order, y up.
    # Returns the new corners and the angle (radians) that was removed.
    p = np.asarray(points, dtype=float)
    if len(p) > 2 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    p = p[np.hypot(*(p - np.roll(p, 1, axis=0)).T) > 1e-6]
    d = np.roll(p, -1, axis=0) - p
    length, angle = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])
    theta = np.angle(np.sum(length * np.exp(4j * angle))) / 4
    c, s = np.cos(theta), np.sin(theta)
    q = p @ np.array([[c, -s], [s, c]])
    d = np.roll(q, -1, axis=0) - q
    horizontal = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    if horizontal.all() or not horizontal.any():
        raise ValueError("at least 4 corners are needed")
    start = int(np.argmax(horizontal != np.roll(horizontal, 1)))
    lines = []
    for e in np.roll(np.arange(len(q)), -start):
        middle = (q[e] + q[(e + 1) % len(q)]) / 2
        position = middle[1] if horizontal[e] else middle[0]
        weight = np.hypot(*d[e]) + 1e-9
        if lines and lines[-1][0] == horizontal[e]:
            lines[-1][1] += weight
            lines[-1][2] += weight * position
        else:
            lines.append([horizontal[e], weight, weight * position])
    lines = [(h, total / w) for h, w, total in lines]
    if len(lines) < 4:
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), theta
    corners = [(v2, v1) if h1 else (v1, v2)
               for (h1, v1), (h2, v2) in zip(lines, lines[1:] + lines[:1])]
    return np.array(corners), theta


def _polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def photo_to_outline(gray, detail=0.012, max_gap=0.18, debug=False):
    # A photo of a hand-drawn silhouette -> its corners, squared (y up, photo pixels).
    # detail: how small a step of the outline is kept (share of the outline's length); max_gap: the largest gap
    # in the drawing that is closed (share of the photo's size). Returns None when no closed shape is found.
    ink, flat = _ink(gray)
    region, links = None, []
    if ink.any():
        closed, links = _close_gaps(ink, max_gap * max(gray.shape))
        region = _enclosed(closed)
        if region is None:                                       # last try: thicker lines
            k = int(0.03 * max(gray.shape)) | 1
            closed = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
            region = _enclosed(closed)
    if region is None:
        return (None, {"flat": flat, "ink": ink, "links": links}) if debug else None
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, detail * cv2.arcLength(c, True), True)[:, 0, :].astype(float)
    corners_y_up = approx * [1, -1]
    squared, theta = square_up(corners_y_up)
    if debug:
        return squared, {"flat": flat, "ink": ink, "links": links, "region": region, "approx": approx, "theta": theta}
    return squared


def outline_from_taps(points_px, square=True):
    # the fallback: the user taps the corners on the photo (pixels, y down) -> squared corners (y up)
    p = np.asarray(points_px, dtype=float) * [1, -1]
    return square_up(p)[0] if square else p


def silhouette_input(corners, area_m2, size=IMG, draw=512, margin=0.05):
    # Squared corners (any unit, y up) + the area the user typed -> the model's input, as in training:
    # turned so the long side is horizontal, scaled to fit with a 5% margin, centred, filled, shrunk to 128 x 128.
    p = np.asarray(corners, dtype=float)
    p = p * np.sqrt(area_m2 / _polygon_area(p))                      # metres
    w, h = np.ptp(p, axis=0)
    if h > w:                                                        # standing: turn 90 degrees, as in training
        p = p @ np.array([[0.0, 1.0], [-1.0, 0.0]])
        w, h = h, w
    p = p - p.min(0)
    scale = draw * (1 - 2 * margin) / max(w, h)
    canvas = np.c_[scale * p[:, 0] + (draw - w * scale) / 2, -scale * p[:, 1] + (draw + h * scale) / 2]
    big = np.zeros((draw, draw), np.uint8)
    cv2.fillPoly(big, [np.round(canvas * 16).astype(np.int32)], 255, shift=4)
    sil = cv2.resize(big.astype(np.float32) / 255, (size, size), interpolation=cv2.INTER_AREA)
    return {"sil_u8": (sil * 255).round().astype(np.uint8), "m_per_px": (draw / size) / scale,
            "corners_m": p, "area_m2": float(area_m2)}


# ============================================================================================
# The whole app: photo (or tapped corners) + area + bedrooms + balcony -> 5 options
# ============================================================================================
if torch is not None:
    def design_from_outline(model, corners, area_m2, bedrooms, balcony, rules, area_mean, area_std,
                            device="cpu", n_generate=30, seed=0, n_options=5, settings=SETTINGS):
        # squared corners (y up) -> model input -> n_generate pictures -> step 3 -> the options
        inp = silhouette_input(corners, area_m2)
        cond = condition_vector(bedrooms, balcony, area_m2, area_mean, area_std)
        maps = generate(model, inp["sil_u8"], cond, n=n_generate, seed=seed, device=device)
        options, reasons = design_from_maps(maps, inp["sil_u8"], bedrooms, balcony, inp["m_per_px"], rules,
                                            n_options=n_options, settings=settings)
        return {"options": options, "reasons": reasons, **inp}

    def design_from_photo(model, source, area_m2, bedrooms, balcony, rules, area_mean, area_std, **kw):
        # a photo of the silhouette (path or uploaded bytes) -> the options; if no closed shape is found,
        # the app asks the user to tap the corners (then use design_from_outline with outline_from_taps)
        corners = photo_to_outline(read_photo(source))
        if corners is None:
            return {"options": [], "error": "no closed shape found in the photo: please tap the corners"}
        result = design_from_outline(model, corners, area_m2, bedrooms, balcony, rules, area_mean, area_std, **kw)
        result["corners_photo"] = corners
        return result
