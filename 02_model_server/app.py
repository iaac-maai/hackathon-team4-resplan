"""Team 4 model server: floor-plan search on ResPlan (Gradio).

Two API endpoints, each with a page for people:
- `/search`  (ours): bedrooms + floor area in sqft -> the 6 closest plans, each with its
  room counts, 8 x 8 relationship matrix and quality scores. Uses `search_index.npz`
  (built in 01_autoencoder/notebook/03_make_tensors.ipynb, see data/01_data_and_analysis.md).
- `/predict` (class example): the Fashion-MNIST anomaly autoencoder from the course,
  kept until our own ONNX model is swapped in.

Runs locally (`python app.py`) and on Render.
"""
import base64
import io
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
N_RESULTS = 6

# ---- Plan search -------------------------------------------------------------
INDEX = dict(np.load(HERE / "search_index.npz"))
ROOM_TYPES = [str(t) for t in INDEX["room_types"]]   # order of room_counts and of the matrix rows/columns
HAS_AREA = ~np.isnan(INDEX["area_sqft"])


def plan_record(i: int, sqft: float) -> dict:
    """One plan as plain JSON: identity, size, rooms, relationship matrix, scores."""
    area = float(INDEX["area_sqft"][i])
    return {
        "id": int(INDEX["ids"][i]),
        "bedrooms": int(INDEX["bedrooms"][i]),
        "bathrooms": int(INDEX["bathrooms"][i]),
        "area_sqft": round(area, 1),
        "area_diff_sqft": round(area - sqft, 1),
        "room_counts": dict(zip(ROOM_TYPES, INDEX["room_counts"][i].tolist())),
        "relation": INDEX["relation"][i].tolist(),     # relation[a][b] = connections between type a and type b
        "scores": {
            "circulation_score": round(float(INDEX["circulation_score"][i]), 3),
            "daylight_score": round(float(INDEX["daylight_score"][i]), 3),
            "cross_vent_rooms": int(INDEX["cross_vent_rooms"][i]),
            "dark_rooms": int(INDEX["dark_rooms"][i]),
            "missing_door": bool(INDEX["missing_door"][i]),
        },
    }


def find_plans(bedrooms: int, sqft: float) -> list[int]:
    """Indices of the N_RESULTS plans closest in area, preferring the exact bedroom count
    and plans where every room is reachable. Relaxes the filters only if too few plans match."""
    diff = np.abs(INDEX["area_sqft"] - sqft)
    beds = INDEX["bedrooms"].astype(int)
    tiers = [
        HAS_AREA & (beds == bedrooms) & ~INDEX["missing_door"],
        HAS_AREA & (beds == bedrooms),
        HAS_AREA & (np.abs(beds - bedrooms) == 1),
        HAS_AREA,
    ]
    chosen: list[int] = []
    for mask in tiers:
        candidates = np.flatnonzero(mask & ~np.isin(np.arange(len(diff)), chosen))
        chosen += candidates[np.argsort(diff[candidates])][: N_RESULTS - len(chosen)].tolist()
        if len(chosen) == N_RESULTS:
            break
    return chosen


def search(bedrooms: int, sqft: float) -> dict:
    """Find 6 floor plans. Send the number of bedrooms and the floor area in square feet.
    Returns the query, the room-type order of the relation matrix, and 6 plans."""
    bedrooms, sqft = int(bedrooms), float(sqft)
    if not 0 <= bedrooms <= 10 or not 100 <= sqft <= 10000:
        raise gr.Error("Use 0-10 bedrooms and 100-10000 sqft.")
    plans = [plan_record(i, sqft) for i in find_plans(bedrooms, sqft)]
    return {
        "query": {"bedrooms": bedrooms, "sqft": sqft},
        "room_types": ROOM_TYPES,
        "count": len(plans),
        "plans": plans,
    }


def search_for_page(bedrooms, sqft):
    result = search(bedrooms, sqft)
    rows = [[p["id"], p["bedrooms"], p["bathrooms"], p["area_sqft"], p["area_diff_sqft"],
             p["scores"]["daylight_score"], p["scores"]["circulation_score"],
             p["scores"]["dark_rooms"], p["scores"]["missing_door"]] for p in result["plans"]]
    return rows, result


# ---- Class example: anomaly autoencoder (Fashion-MNIST) ----------------------
CONFIG = json.loads((HERE / "anomaly_ae_config.json").read_text())
SIZE = CONFIG["image_size"]
THRESHOLD = CONFIG["threshold"]
session = ort.InferenceSession(HERE / "anomaly_ae.onnx")
INPUT_NAME = session.get_inputs()[0].name


def preprocess(image: Image.Image):
    grey = image.convert("L").resize((SIZE, SIZE))
    x = np.asarray(grey, dtype=np.float32) / 255.0
    inverted = bool(x.mean() > 0.5)
    if inverted:
        x = 1.0 - x
    return x[None, None], inverted


def to_image(array, scale=8):
    img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    return img.resize((SIZE * scale, SIZE * scale), Image.NEAREST)


def analyse(image: Image.Image):
    x, inverted = preprocess(image)
    rebuilt = session.run(None, {INPUT_NAME: x})[0]
    error_map = ((rebuilt - x) ** 2)[0, 0]
    score = float(error_map.mean())
    return {
        "score": round(score, 5),
        "threshold": round(THRESHOLD, 5),
        "is_anomaly": score > THRESHOLD,
        "score_vs_threshold": round(score / THRESHOLD, 2),
        "inverted_input": inverted,
    }, rebuilt[0, 0], error_map


def analyse_for_page(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    result, rebuilt, error_map = analyse(image)
    verdict = "Anomaly" if result["is_anomaly"] else "Normal"
    return {verdict: 1.0}, result, to_image(rebuilt), to_image(error_map / max(error_map.max(), 1e-8))


def predict(image_b64: str) -> dict:
    """Class example. Score one image: send a base64-encoded PNG or JPEG."""
    if "," in image_b64[:100]:
        image_b64 = image_b64.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    result, rebuilt, _ = analyse(image)
    buffer = io.BytesIO()
    to_image(rebuilt, scale=4).save(buffer, format="PNG")
    result["reconstruction_png_base64"] = base64.b64encode(buffer.getvalue()).decode()
    return result


# ---- Page ----------------------------------------------------------------------
with gr.Blocks(title="Team 4 · ResPlan search") as demo:
    gr.Markdown("# Team 4 · ResPlan floor-plan search\n"
                f"{len(INDEX['ids']):,} cleaned plans from ResPlan. "
                "Give a bedroom count and a floor area: get the 6 closest plans with their scores.")
    with gr.Tab("Search plans"):
        with gr.Row():
            bedrooms_in = gr.Number(value=3, precision=0, label="Bedrooms")
            sqft_in = gr.Number(value=1100, label="Floor area (sqft)")
        search_btn = gr.Button("Search", variant="primary")
        table_out = gr.Dataframe(headers=["id", "bedrooms", "bathrooms", "area_sqft", "diff_sqft",
                                          "daylight", "circulation", "dark_rooms", "missing_door"],
                                 label="6 plans")
        json_out = gr.JSON(label="API answer (with relation matrices)")
        search_btn.click(search_for_page, inputs=[bedrooms_in, sqft_in], outputs=[table_out, json_out],
                         api_visibility="private")
    with gr.Tab("Class example: anomaly detector"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Input image")
                button = gr.Button("Analyse", variant="primary")
                gr.Examples(sorted(str(p) for p in (HERE / "examples").glob("*.png")), inputs=image_in)
            with gr.Column():
                verdict_out = gr.Label(label="Verdict")
                details_out = gr.JSON(label="Details")
                with gr.Row():
                    rebuilt_out = gr.Image(label="Reconstruction")
                    error_out = gr.Image(label="Error map")
        button.click(analyse_for_page, inputs=image_in, outputs=[verdict_out, details_out, rebuilt_out, error_out],
                     api_visibility="private")
    gr.api(search, api_name="search")
    gr.api(predict, api_name="predict")

if __name__ == "__main__":
    # Render tells the app which port to use in the PORT variable; locally it is 7860
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
