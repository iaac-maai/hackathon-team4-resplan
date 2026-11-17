"""app.py - Team 4 layout generator: the web server (the backend of the app).

    GET  /                    the test page
    GET  /api                 how to use the API (JSON)
    GET  /health              is the server ready?
    POST /api/design          photo of the silhouette + area + bedrooms + balcony  ->  5 options
    POST /api/design/outline  corners tapped on the photo + area + bedrooms + balcony  ->  5 options

Run it on your computer:   python app.py      then open http://localhost:7860
On Render it starts by itself (see render.yaml at the root of the repository).
"""
import base64
import contextlib
import os
import traceback
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import service

HERE = Path(__file__).resolve().parent
MAX_PHOTO_MB = 15

API = {
    "POST /api/design": {
        "send as": "multipart/form-data, or JSON with the photo as text: "
                   "{photo_base64, area_m2, bedrooms, balcony}",
        "fields": {
            "photo": "the photo of the hand-drawn silhouette (JPEG or PNG file)",
            "area_m2": "the area of the whole silhouette in m2 (balcony included), for example 95",
            "bedrooms": "number of bedrooms, 1 to 5",
            "balcony": "1 or 0 (also true/false)",
            "seed": "optional, a whole number: another seed gives other options (default 0)",
            "dxf": "optional, 0 to leave out the DXF files (faster, smaller answer)",
            "preview": "optional, 0 to leave out the PNG pictures",
        },
        "if no closed shape is found": "the answer has ok=false and need_taps=true: show the photo, let the user "
                                      "tap the corners, and send them to /api/design/outline",
    },
    "POST /api/design/outline": {
        "send as": "application/json",
        "body": {"points": "[[x, y], ...] the corners tapped on the photo, in order, in pixels, y down",
                 "area_m2": 95, "bedrooms": 3, "balcony": 1, "seed": 0},
    },
    "answer": {
        "ok": "true when there is at least one option",
        "options": "up to 5, best first. Each: score (0-1), summary, rooms [{type, area_m2, polygons}], walls, doors, "
                   "windows, rails, entrance, outline, preview_png (data URL), dxf_base64",
        "polygons": "a polygon is a list of rings (the first is the outside, the next ones are holes); "
                    "a ring is a list of [x, y] points",
        "units": "metres, y up, (0, 0) = bottom-left corner of the flat",
        "outline": "the silhouette that was used: metres, and photo_px (on the photo, pixels, y down)",
        "pictures": "how many pictures the model drew, how many could be used, and why the others were dropped",
        "seconds": "time on the server",
    },
}


@contextlib.asynccontextmanager
async def lifespan(app):
    if getattr(app.state, "designer", None) is None:          # load the model once, when the server starts
        app.state.designer = service.Designer()
    yield


def _error(message, status=400):
    return JSONResponse({"ok": False, "error": message}, status_code=status)


async def index(request):
    return FileResponse(HERE / "static" / "index.html")


async def api_help(request):
    return JSONResponse(API)


async def health(request):
    return JSONResponse({"ok": True, **request.app.state.designer.info()})


def _flag(value, default=True):
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


async def design_photo(request):
    if request.headers.get("content-type", "").startswith("application/json"):
        try:                                           # the photo as text (base64) inside JSON
            form = await request.json()
            text = str(form.get("photo_base64") or "")
            data = base64.b64decode(text.split(",", 1)[1] if text.startswith("data:") else text)
        except Exception:
            return _error("send JSON: {photo_base64, area_m2, bedrooms, balcony}")
        if not data:
            return _error("the field 'photo_base64' is missing, or it is not a photo in base64")
    else:
        try:
            form = await request.form()
        except Exception:
            return _error("send the fields as multipart/form-data (photo, area_m2, bedrooms, balcony)")
        photo = form.get("photo")
        if photo is None or isinstance(photo, str):
            return _error("the field 'photo' must be a file")
        data = await photo.read()
    if len(data) > MAX_PHOTO_MB * 1024 * 1024:
        return _error(f"the photo is larger than {MAX_PHOTO_MB} MB", 413)
    for name in ["area_m2", "bedrooms"]:
        if form.get(name) in (None, ""):
            return _error(f"the field '{name}' is missing")
    return await _run(request.app.state.designer.from_photo, data, form.get("area_m2"), form.get("bedrooms"),
                      form.get("balcony", "0"), int(form.get("seed") or 0), _flag(form.get("dxf")), _flag(form.get("preview")))


async def design_outline(request):
    try:
        body = await request.json()
    except Exception:
        return _error("send a JSON body: {points, area_m2, bedrooms, balcony}")
    for name in ["points", "area_m2", "bedrooms"]:
        if body.get(name) in (None, ""):
            return _error(f"the field '{name}' is missing")
    return await _run(request.app.state.designer.from_taps, body["points"], body["area_m2"], body["bedrooms"],
                      body.get("balcony", 0), int(body.get("seed") or 0), _flag(body.get("dxf")), _flag(body.get("preview")))


async def _run(fn, *args):
    try:
        result = await run_in_threadpool(fn, *args)
    except service.InputError as e:
        return _error(str(e))
    except Exception as e:
        traceback.print_exc()
        return _error(f"server error: {type(e).__name__}: {e}", 500)
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api", api_help),
        Route("/health", health),
        Route("/api/design", design_photo, methods=["POST"]),
        Route("/api/design/outline", design_outline, methods=["POST"]),
        Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
    ],
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn
    # Render gives the port in $PORT; on your computer it is 7860
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
