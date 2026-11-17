# 02 · Model server

**Public URL:** https://team4-resplan-server.onrender.com *(check after the first Render deploy)*
**API page:** https://team4-resplan-server.onrender.com/?view=api

> ⚠️ The free Render service **sleeps after 15 minutes** without requests. The first call then takes about **one minute** while it wakes up.

A Gradio server (page for people + API for programs) on Render, free plan, region Frankfurt. Render reads `render.yaml` at the repo root (`rootDir: 02_model_server`).

## Endpoints

### `/search`: our floor-plan search

| | |
|---|---|
| **Input** | `bedrooms` (int, 0–10), `sqft` (floor area in square feet, 100–10000) |
| **Output** | JSON: the query, `room_types` (the order of the matrix rows and columns), and **6 plans** |
| **Data** | `search_index.npz`: 16,317 cleaned ResPlan plans (see [`data/01_data_and_analysis.md`](../data/01_data_and_analysis.md)) |

How the 6 plans are chosen: the closest floor area among plans with **exactly** that bedroom count where every room is reachable through doors. If fewer than 6 match, the server also takes plans with a missing door, then ±1 bedroom, then any plan.

Each plan:

```json
{
  "id": 15968,
  "bedrooms": 3, "bathrooms": 2,
  "area_sqft": 1100.0, "area_diff_sqft": 0.0,
  "room_counts": {"living": 1, "bedroom": 3, "kitchen": 1, "bathroom": 2, "balcony": 0, "storage": 0, "stair": 0, "front_door": 1},
  "relation": [[0, 3, 1, 2, 0, 0, 0, 1], ...],
  "scores": {"circulation_score": 0.857, "daylight_score": 0.6, "cross_vent_rooms": 0, "dark_rooms": 0, "missing_door": false}
}
```

- `relation[a][b]`: the number of connections between room type `a` and room type `b` (order: living, bedroom, kitchen, bathroom, balcony, storage, stair, front_door).
- `circulation_score`: the share of rooms within 2 steps of the entrance (0–1, higher = more compact).
- `daylight_score`: the share of living rooms, bedrooms and kitchens with daylight (0–1).
- `cross_vent_rooms`: rooms with windows on opposite walls. `dark_rooms`: living rooms, bedrooms and kitchens with no window and no balcony.
- `missing_door`: true if a room cannot be reached through doors.

### `/predict`: class example (placeholder)

This is the course's Fashion-MNIST anomaly autoencoder (`anomaly_ae.onnx`), kept unchanged from `hackathon/docs/model_app/`. **It is not our model.** Our own ResPlan autoencoder will replace it here.

## API example (copy and run)

```python
# pip install gradio_client
from gradio_client import Client

client = Client("https://team4-resplan-server.onrender.com")
result = client.predict(3, 1100, api_name="/search")   # bedrooms, sqft
for p in result["plans"]:
    print(p["id"], p["area_sqft"], p["scores"])
```

Plain HTTP (two steps, as the Cloudflare app will do it):

```bash
EVENT=$(curl -s -X POST https://team4-resplan-server.onrender.com/gradio_api/call/search \
  -H "Content-Type: application/json" -d '{"data": [3, 1100]}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['event_id'])")
curl -N https://team4-resplan-server.onrender.com/gradio_api/call/search/$EVENT
```

[`call_api.ipynb`](call_api.ipynb) runs all of this and prints the result.

## Run locally

```bash
cd 02_model_server
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
```

Before each push, check that `/search` still answers in the agreed format:

```bash
python tests/test_search.py     # prints "all 3 tests passed"
```

## Files

| File | Contents |
|---|---|
| `app.py` | The search, the class example, the page and the API |
| `search_index.npz` | Plan metadata, relationship matrices and scores |
| `anomaly_ae.onnx`, `anomaly_ae_config.json`, `examples/` | The class example model |
| `requirements.txt`, `.python-version` | What Render needs (Python 3.12, pinned packages) |
| `call_api.ipynb` | Calls the live API and prints the result |
| `tests/test_search.py` | Checks the `/search` answer format (6 plans, 8 × 8 matrix, all scores) |

## Limitations

- The search is a filter on metadata. It does not use the autoencoder yet.
- Plan images are not served yet: use `id` to find `plans_png/<id>.png` (team Google Drive).
- All plans are South Asian flats. There is no north direction in the data. ResPlan licence: non-commercial use, credit ResPlan.
