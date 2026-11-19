# Team 4 · Layout generator: the model server

IAAC MaAI · AI Methods hackathon, November 2026 · Team 4.

The user draws the silhouette of a flat, takes a photo, and types three things: the **area** (m²), the number of **bedrooms**, and **balcony** yes/no. The server answers with **5 floor plans**, as vectors in metres and as DXF files.

How: the photo becomes the silhouette (step 4). Then our model, a conditional VAE trained on ResPlan and exported to ONNX, draws 30 room maps for this silhouette. Step 3 turns them into clean plans with walls, doors and windows, keeps the ones that follow the rules, and chooses the 5 best that are different from each other. The code is in `layout_core.py`, the same file the notebooks use.

```text
02_server/
├── app.py              the server: the API and the test page
├── service.py          one request: photo → silhouette → model → 5 plans
├── layout_core.py      steps 2-4 (a copy, made by publish.py)
├── model/              layout_decoder.onnx + settings (made by publish.py)
├── static/             the test page and the example sketch
├── requirements.txt    the packages Render installs
├── .python-version     Python 3.12
└── publish.py          run it on your computer before git push
render.yaml             (at the root of the repository) the Render Blueprint
```

## Put it online on Render

1. **On a computer with the notebooks' environment:** `cd 02_server` then `python publish.py`. It copies the latest `layout_core.py`, turns the model into `model/layout_decoder.onnx`, and checks it.
2. **Push to GitHub:** the `02_server` folder and `render.yaml`.
3. **render.com:** New → **Blueprint** → this repository. Render reads `render.yaml`: a free web service `team4-layout-server` in Frankfurt, built from the `02_server` folder.
4. **Wait for "Live"** (the build log shows each step). Open `https://team4-layout-server.onrender.com`: the test page.

The free server is small (512 MB, 0.1 CPU): a request takes much longer than on a laptop. It sleeps after 15 minutes without use; the first request wakes it up (about a minute). If the build fails, read the build log first.

## The API (for the app)

| | |
|---|---|
| `POST /api/design` | a photo + area + bedrooms + balcony → 5 options |
| `POST /api/design/outline` | the corners the user tapped on the photo + the same numbers → 5 options |
| `GET /health` | is the server ready? |
| `GET /api` | this description, as JSON |

### 1. From a photo

Fields:

| field | | |
|---|---|---|
| `photo` | file | the photo of the drawing (JPEG or PNG) |
| `area_m2` | number | the whole silhouette, balcony included (25 to 400) |
| `bedrooms` | 1 to 5 | |
| `balcony` | 1 or 0 | |
| `seed` | optional | another number gives other options (default 0) |
| `dxf`, `preview` | optional | 0 to leave out the DXF files or the PNG pictures (smaller and faster answer) |

**Taking the photo with the camera** (in the app's own page): on a phone, `<input type="file" accept="image/*" capture="environment">` opens the phone's camera, and the photo comes back like a chosen file. On a computer, the webcam can be opened with `navigator.mediaDevices.getUserMedia` (the page must be on `https` or `localhost`). The test page does both: see "Take a photo" in `static/index.html`.

As a form (`multipart/form-data`), from a notebook:

```python
import requests
url = "https://team4-layout-server.onrender.com/api/design"
with open("sketch.jpg", "rb") as f:
    answer = requests.post(url, files={"photo": f}, data={"area_m2": 95, "bedrooms": 3, "balcony": 1}, timeout=300).json()
print(answer["ok"], len(answer["options"]), answer["seconds"])
```

Or as JSON, with the photo as text (base64), for example from a Cloudflare Worker:

```js
const answer = await (await fetch("https://team4-layout-server.onrender.com/api/design", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ photo_base64: photoBase64, area_m2: 95, bedrooms: 3, balcony: 1 }),
})).json();
```

### 2. If no closed shape is found: tap the corners

The answer is `{"ok": false, "need_taps": true, ...}`. Show the photo, let the user tap the corners **in order, all the way around**, and send them:

```js
await fetch("https://team4-layout-server.onrender.com/api/design/outline", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ points: [[60, 70], [835, 75], [830, 520], [65, 515]], area_m2: 95, bedrooms: 3, balcony: 1 }),
});
```

`points` are pixels on the photo (x to the right, y down), as the browser gives them.

### The answer

```text
ok            true when there is at least one option
options       up to 5, best first
  rank, score (0-1), score_parts
  summary     {bedrooms, bathrooms, kitchens, balconies, rooms_m2, balcony_m2, kitchen_rescued}
  rooms       [{type, area_m2, polygons}]   type: living, bedroom, bathroom, kitchen, balcony
  walls, doors, windows, rails, entrance, outline     polygons
  preview_png a picture of the plan (data URL: put it in <img src>)
  dxf_base64  the DXF file (base64)
outline       the silhouette that was used: metres, and photo_px (on the photo, to draw it over the photo)
pictures      how many pictures the model drew, how many could be used, and why the others were dropped
seconds       time on the server
error         a sentence to show to the user, when something is wrong
```

- **Units:** metres, **y up**, (0, 0) = the bottom-left corner of the flat. In SVG (y down) draw with `transform="scale(1,-1)"`, or use `y = height - y`.
- **Polygon:** a list of rings. The first ring is the outside, the next ones are holes. A ring is a list of `[x, y]` points.
- Room areas are **net**: inside the walls.
- Wrong values (for example 9 bedrooms) get HTTP 400 and `{"ok": false, "error": "..."}`. The server accepts calls from any website (CORS is open).

## Run it on your computer

In the same Python environment as the notebooks (it already has PyTorch, OpenCV, scikit-learn...):

```bash
cd 02_server
python -m pip install starlette uvicorn python-multipart ezdxf onnxruntime onnx
python publish.py
python app.py            # then open http://localhost:7860 (Ctrl+C to stop)
```

Do not install `requirements.txt` in the notebooks' environment: its `opencv-python-headless` clashes with the `opencv-python` that is already there. `requirements.txt` is for Render.

## Good to know

- One request is answered at a time. If fewer than 5 options are found in 30 pictures, the server draws 30 more. Settings (environment variables): `N_GENERATE` (30), `MAX_ROUNDS` (2), `CHUNK` (10 pictures at a time, to save memory), `THREADS` (1).
- The rules (wall thickness, smallest bedroom, one kitchen...) are in `SETTINGS` at the top of `layout_core.py`. After a change, run `publish.py` again and push.
- The server draws its own random numbers, so `seed=0` does not give the same plans as the notebooks.
- `Dockerfile` is only for hosts that run Docker; Render does not use it.
- The plans are invented by a model: they are a starting point for the architect, not a checked design.
