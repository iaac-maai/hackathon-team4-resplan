"""service.py - Team 4 layout generator: one request of the app, from a photo (or tapped corners) to 5 options.

The model is loaded once (class Designer). app.py (the web server) calls Designer.from_photo / Designer.from_taps.
Everything here can also be used directly in Python, without the server.
"""
import base64
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import layout_core as lc

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "model"))
ONNX_FILE = "layout_decoder.onnx"                          # made by publish.py from the PyTorch weights
N_GENERATE = int(os.environ.get("N_GENERATE", 30))       # pictures the model draws per round
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 2))         # fewer than 5 options after one round: one more round
CHUNK = int(os.environ.get("CHUNK", 10))                  # pictures drawn at the same time (less memory)
THREADS = int(os.environ.get("THREADS", 1))               # CPU threads for the model (the free Render server has 0.1 CPU)
AREA_RANGE = (25.0, 400.0)                                 # m2 accepted (the training flats: about 60-270 m2)
LAYOUT_KEYS = ["walls", "doors", "windows", "rails", "entrance", "outline"]


class InputError(ValueError):
    """A wrong value sent by the app (answered with HTTP 400)."""


def check_inputs(area_m2, bedrooms, balcony):
    try:
        area_m2, bedrooms = float(area_m2), int(bedrooms)
    except (TypeError, ValueError):
        raise InputError("area_m2 must be a number and bedrooms a whole number")
    if not AREA_RANGE[0] <= area_m2 <= AREA_RANGE[1]:
        raise InputError(f"area_m2 must be between {AREA_RANGE[0]:.0f} and {AREA_RANGE[1]:.0f}")
    if not 1 <= bedrooms <= lc.BED_MAX:
        raise InputError(f"bedrooms must be between 1 and {lc.BED_MAX}")
    if isinstance(balcony, str):
        balcony = balcony.strip().lower() in ("1", "true", "yes", "on")
    return area_m2, bedrooms, int(bool(balcony))


def read_photo_bytes(data):
    # uploaded bytes -> grey image with the long side lc.PHOTO_SIZE, and the factor back to the photo's own pixels.
    # The phone's rotation (EXIF) is applied, like the browser does when it shows the photo.
    from PIL import Image, ImageOps
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    except Exception:
        raise InputError("cannot read the photo: send a JPEG or PNG file")
    gray = np.asarray(img)
    k = lc.PHOTO_SIZE / max(gray.shape)
    gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return gray, 1 / k, img.size


def onnx_generate(session, latent_dim):
    # the same as lc.generate, with ONNX Runtime instead of PyTorch (much smaller on a server)
    def generate(model, sil_u8, cond, n=30, seed=None, device="cpu"):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, latent_dim)).astype(np.float32)
        s = np.repeat((np.asarray(sil_u8, dtype=np.float32) / 255)[None, None], n, axis=0)
        c = np.repeat(np.asarray(cond, dtype=np.float32)[None], n, axis=0)
        parts = [session.run(None, {"z": z[i:i + CHUNK], "s": s[i:i + CHUNK], "c": c[i:i + CHUNK]})[0]
                 for i in range(0, n, CHUNK)]
        return np.concatenate(parts)
    return generate


def _map_rings(layout, fn):
    # apply fn to every ring (n, 2) of a layout; returns a new layout
    out = dict(layout)
    out["rooms"] = [dict(r, polygons=[[fn(ring) for ring in poly] for poly in r["polygons"]]) for r in layout["rooms"]]
    for key in LAYOUT_KEYS:
        out[key] = [[fn(ring) for ring in poly] for poly in layout.get(key, [])]
    return out


def to_user_frame(layout, turned):
    # The model works with the long side horizontal. Turn the plan back to the user's drawing,
    # and put the bottom-left corner of the flat at (0, 0). Metres, y up.
    rings = [ring for poly in layout["outline"] for ring in poly] or \
            [ring for r in layout["rooms"] for poly in r["polygons"] for ring in poly]
    low = np.min([ring.min(0) for ring in rings], axis=0)
    turn = (lambda p: np.c_[p[:, 1], -p[:, 0]]) if turned else (lambda p: p)
    moved = _map_rings(layout, lambda ring: turn(np.asarray(ring) - low))
    rings = [ring for poly in moved["outline"] for ring in poly] or \
            [ring for r in moved["rooms"] for poly in r["polygons"] for ring in poly]
    low = np.min([ring.min(0) for ring in rings], axis=0)
    return _map_rings(moved, lambda ring: np.round(np.asarray(ring) - low, 4))


def _lists(polys):
    return [[np.round(ring, 3).tolist() for ring in poly] for poly in polys]


def _png(fig_or_array):
    if isinstance(fig_or_array, np.ndarray):
        ok, buf = cv2.imencode(".png", fig_or_array)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    buf = io.BytesIO()
    fig_or_array.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def preview_png(layout):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111)
    lc.draw_layout(layout, ax, "")
    return _png(fig)


def dxf_base64(layout, note):
    try:
        import ezdxf  # noqa: F401
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "option.dxf"
        lc.export_dxf(layout, path, note=note)
        return base64.b64encode(path.read_bytes()).decode()


def option_json(layout, rank, request, with_dxf=True, with_preview=True):
    counts = {t: sum(r["type"] == t for r in layout["rooms"]) for t in lc.ROOMS}
    area = {t: round(sum(r["area_m2"] for r in layout["rooms"] if r["type"] == t), 1) for t in lc.ROOMS}
    out = {
        "rank": rank,
        "score": round(float(layout["score"]), 3),
        "score_parts": {k: round(float(v), 3) for k, v in layout["score_parts"].items()},
        "summary": {"bedrooms": counts["bedroom"], "bathrooms": counts["bathroom"], "kitchens": counts["kitchen"],
                    "balconies": counts["balcony"], "rooms_m2": round(sum(v for k, v in area.items() if k != "balcony"), 1),
                    "balcony_m2": area["balcony"], "kitchen_rescued": bool(layout.get("kitchen_boost", 1.0) > 1)},
        "rooms": [{"type": r["type"], "area_m2": r["area_m2"], "polygons": _lists(r["polygons"])} for r in layout["rooms"]],
        **{key: _lists(layout.get(key, [])) for key in LAYOUT_KEYS},
        "doors_info": layout.get("doors_info", []),
    }
    if with_preview:
        out["preview_png"] = preview_png(layout)
    if with_dxf:
        note = (f"Team 4 layout generator - {request['area_m2']:g} m2, {request['bedrooms']} bedrooms, "
                f"balcony {'yes' if request['balcony'] else 'no'} - option {rank}")
        out["dxf_base64"] = dxf_base64(layout, note)
    return out


class Designer:
    def __init__(self, model_dir=MODEL_DIR, device="cpu", generate=None):
        # generate: only for tests without PyTorch (a function like lc.generate)
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "layout_cvae_config.json").read_text())
        self.rules = json.loads((model_dir / "room_rules.json").read_text())
        self.area_mean = self.config["condition"]["area_mean"]
        self.area_std = self.config["condition"]["area_std"]
        self.device, self.model = device, None
        self.lock = threading.Lock()                   # one request at a time (PyTorch and matplotlib share memory)
        if generate is not None:
            self._generate = generate
            return
        import onnxruntime as ort
        path = model_dir / ONNX_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run publish.py first (it makes it from the PyTorch weights)")
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self._generate = onnx_generate(session, self.config["latent_dim"])

    def info(self):
        return {"model": "LayoutCVAE decoder (Team 4, trained on ResPlan), ONNX", "device": self.device,
                "bedrooms": [1, lc.BED_MAX], "area_m2": list(AREA_RANGE), "pictures_per_round": N_GENERATE,
                "settings": {k: v for k, v in lc.SETTINGS.items() if not isinstance(v, dict)}}

    def from_photo(self, photo, area_m2, bedrooms, balcony, seed=0, dxf=True, preview=True):
        # photo: the bytes of the uploaded file
        area_m2, bedrooms, balcony = check_inputs(area_m2, bedrooms, balcony)
        start = time.time()
        gray, to_photo, size = read_photo_bytes(photo)
        corners, dbg = lc.photo_to_outline(gray, debug=True)
        if corners is None:
            return {"ok": False, "need_taps": True, "photo_size": list(size),
                    "error": "No closed shape was found in the photo. Please tap the corners of the flat, in order."}
        c, s = np.cos(dbg["theta"]), np.sin(dbg["theta"])
        on_photo = corners @ np.array([[c, -s], [s, c]]).T * [1, -1] * to_photo     # the squared outline, on the photo
        result = self._design(corners, area_m2, bedrooms, balcony, seed, dxf, preview)
        result["outline"]["photo_px"] = np.round(on_photo, 1).tolist()
        result["outline"]["source"] = "photo"
        result["photo_size"] = list(size)
        result["seconds"] = round(time.time() - start, 2)
        return result

    def from_taps(self, points, area_m2, bedrooms, balcony, seed=0, dxf=True, preview=True):
        # points: the corners the user tapped on the photo, in order: [[x, y], ...] in pixels, y down
        area_m2, bedrooms, balcony = check_inputs(area_m2, bedrooms, balcony)
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4:
            raise InputError("points must be at least 4 corners: [[x, y], [x, y], ...]")
        start = time.time()
        try:
            corners, theta = lc.square_up(p * [1, -1])            # the same as lc.outline_from_taps, with the angle
        except ValueError:
            raise InputError("these corners do not make a closed shape with straight walls: tap at least 4 corners, in order")
        result = self._design(corners, area_m2, bedrooms, balcony, seed, dxf, preview)
        c, s = np.cos(theta), np.sin(theta)
        result["outline"]["photo_px"] = np.round(corners @ np.array([[c, -s], [s, c]]).T * [1, -1], 1).tolist()
        result["outline"]["source"] = "taps"
        result["seconds"] = round(time.time() - start, 2)
        return result

    def _design(self, corners, area_m2, bedrooms, balcony, seed, dxf, preview):
        request = {"area_m2": area_m2, "bedrooms": bedrooms, "balcony": bool(balcony), "seed": int(seed)}
        inp = lc.silhouette_input(corners, area_m2)
        w, h = np.ptp(np.asarray(corners, dtype=float), axis=0)
        turned = bool(h > w)                                     # silhouette_input turned it: turn the plans back
        cond = lc.condition_vector(bedrooms, balcony, area_m2, self.area_mean, self.area_std)
        with self.lock:
            maps = self._generate(self.model, inp["sil_u8"], cond, n=N_GENERATE, seed=int(seed), device=self.device)
            options, reasons = lc.design_from_maps(maps, inp["sil_u8"], bedrooms, balcony, inp["m_per_px"], self.rules)
            rounds = 1
            while len(options) < 5 and rounds < MAX_ROUNDS:
                more = self._generate(self.model, inp["sil_u8"], cond, n=N_GENERATE, seed=int(seed) + 1000 * rounds,
                                      device=self.device)
                maps = np.concatenate([maps, more])
                options, reasons = lc.design_from_maps(maps, inp["sil_u8"], bedrooms, balcony, inp["m_per_px"], self.rules)
                rounds += 1
            options = [to_user_frame(o, turned) for o in options]
            out_options = [option_json(o, k, request, dxf, preview) for k, o in enumerate(options, start=1)]
        user_m = np.asarray(corners, dtype=float) * np.sqrt(area_m2 / lc._polygon_area(np.asarray(corners, dtype=float)))
        user_m = user_m - user_m.min(0)
        result = {
            "ok": bool(out_options),
            "request": request,
            "outline": {"metres": np.round(user_m, 3).tolist()},
            "model_input_png": _png(inp["sil_u8"]) if preview else None,
            "pictures": {"drawn": int(len(maps)), "usable": int(reasons.get("ok", 0)),
                         "dropped": {k: int(v) for k, v in reasons.most_common() if k != "ok"}},
            "options": out_options,
            "units": "metres, y up, (0, 0) = bottom-left corner of the flat",
        }
        if not out_options:
            result["error"] = ("The model could not make a plan that follows the rules for this silhouette and these "
                               "numbers. Try another number of bedrooms, a bigger area, or another seed.")
        return result
