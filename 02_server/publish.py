"""publish.py - get the 02_server folder ready for Render (run it on your computer, before git push).

    python publish.py                  the model in 01_autoencoder/models/layout_generator (v1)
    python publish.py --model layout_generator_v2      another trained model, for example v2 from notebook 07

1. copies layout_core.py, the model settings and the example sketch from 01_autoencoder into this folder
2. turns the PyTorch model into ONNX (model/layout_decoder.onnx): the server runs it without PyTorch
3. checks that the ONNX model gives the same numbers as the PyTorch model

Run it again after every change to layout_core.py or after training the model again,
then commit and push the 02_server folder: Render takes the files from GitHub.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL = REPO / "01_autoencoder" / "models" / "layout_generator"


def copies():
    return {
        REPO / "01_autoencoder" / "notebook" / "layout_core.py": HERE / "layout_core.py",
        MODEL / "layout_cvae_config.json": HERE / "model" / "layout_cvae_config.json",
        MODEL / "room_rules.json": HERE / "model" / "room_rules.json",
        REPO / "01_autoencoder" / "layout_examples" / "sketch_example.jpg": HERE / "static" / "sketch_example.jpg",
    }
ONNX = HERE / "model" / "layout_decoder.onnx"


def copy_files():
    for source, target in copies().items():
        if not source.exists():
            raise SystemExit(f"missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"copied    {target.relative_to(HERE)}")


def export_onnx():
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is needed for this step: run it in the notebooks' environment")
    sys.path.insert(0, str(HERE))
    import layout_core as lc

    config = json.loads((MODEL / "layout_cvae_config.json").read_text())
    model = lc.make_model(config)                       # v1 or v2, as the config says
    model.load_state_dict(torch.load(MODEL / config["weights"], map_location="cpu"))
    model.eval()

    class Decoder(torch.nn.Module):                  # the app needs only the decoder: z + silhouette + condition -> maps
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, z, s, c):
            return self.m.decode(z, s, c)

    n = 3
    z = torch.randn(n, config["latent_dim"])
    s = (torch.rand(n, 1, lc.IMG, lc.IMG) > 0.3).float()
    c = torch.rand(n, lc.BED_MAX + 2)
    options = dict(input_names=["z", "s", "c"], output_names=["maps"], opset_version=17,
                   dynamic_axes={"z": {0: "n"}, "s": {0: "n"}, "c": {0: "n"}, "maps": {0: "n"}})
    try:
        torch.onnx.export(Decoder(model), (z, s, c), str(ONNX), dynamo=False, **options)
    except TypeError:                                 # older PyTorch: no 'dynamo' argument
        torch.onnx.export(Decoder(model), (z, s, c), str(ONNX), **options)
    print(f"exported  {ONNX.relative_to(HERE)}  ({ONNX.stat().st_size / 1e6:.1f} MB)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed: the ONNX file was not checked (pip install onnxruntime)")
        return
    out = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"]).run(
        None, {"z": z.numpy(), "s": s.numpy(), "c": c.numpy()})[0]
    with torch.no_grad():
        ref = model.decode(z, s, c).numpy()
    diff = float(abs(out - ref).max())
    print(f"checked   ONNX vs PyTorch: largest difference {diff:.1e} " + ("(ok)" if diff < 1e-3 else "(TOO BIG: tell Claude)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="layout_generator", help="the model's folder in 01_autoencoder/models")
    MODEL = REPO / "01_autoencoder" / "models" / parser.parse_args().model
    print(f"model: {MODEL.relative_to(REPO)}")
    copy_files()
    export_onnx()
    print("\nready.")
    print("  try it here:  python app.py      then open http://localhost:7860")
    print("  then:         commit and push the 02_server folder (and render.yaml) to GitHub")
