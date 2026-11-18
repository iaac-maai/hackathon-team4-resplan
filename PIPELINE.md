# Layout generator: how a sketch becomes 5 floor plans

Team 4 · IAAC MaAI · AI Methods hackathon, November 2026

The user draws the silhouette of a flat, takes a photo, and types the area, the number of bedrooms and balcony yes/no. The app answers with 5 floor plans (vectors in metres + DXF).

**Purple = learned from data (the model). Green = fixed rules and image processing.**

```mermaid
flowchart LR
    T["<b>Training</b> (offline, once)<br>15,832 ResPlan plans →<br>conditional VAE (encoder + decoder)"]
    A["<b>Input</b><br>photo of a hand sketch<br>+ area (m²)<br>+ bedrooms (1–5)<br>+ balcony yes / no"]
    B["<b>Step 4 · Photo → silhouette</b><br>shadow, pen lines,<br>gaps closed, corners squared<br>→ 128 × 128 mask"]
    C["<b>Step 2 · CVAE decoder</b><br>32 random numbers<br>+ silhouette + condition<br>→ 30 new room maps"]
    D["<b>Step 3 · Rules</b><br>straight rooms, exact bedrooms,<br>walls, doors, windows,<br>drop bad plans, score"]
    E["<b>Output</b><br>5 different plans, best first<br>vectors in metres + DXF"]

    A --> B --> C --> D --> E
    T -.->|"trained weights<br>(decoder only, ONNX)"| C

    classDef model fill:#EEEDFE,stroke:#7F77DD,color:#26215C
    classDef rule fill:#E1F5EE,stroke:#1D9E75,color:#04342C
    classDef plain fill:#FFFFFF,stroke:#B4B2A9,color:#1D1D1B
    class T,C model
    class B,D rule
    class A,E plain
```

## What the model invents (learned from data)

- **The zoning:** where the living room, bedrooms, bathrooms, kitchen and balcony go inside the drawn silhouette.
- **Room sizes** and which rooms sit next to each other.
- **Other random numbers → another plan** for the same silhouette. The "Show other options" button on the test page does this.
- **Nothing is looked up:** the server holds no ResPlan plan, only the model file (`02_server/model/layout_decoder.onnx`, 3 MB). A hand-drawn silhouette is not in the data at all.

## What fixed rules do (not learned)

- **Straight walls:** the zoning is cut into rectangles.
- **The exact number of bedrooms:** the raw model is right 37% of the time (±1 bedroom in 86%), so step 3 splits or rejects.
- **Walls** (20 cm outside, 10 cm inside), **doors, windows, entrance.**
- **Drop the plans that break the rules** (bedroom ≥ 9 m², one kitchen, a main bathroom ≥ 3.5 m²…), then keep the 5 best that differ from each other by at least 25%.

In one sentence: **the model invents the zoning, and the rules turn it into a clean plan.** It is not a search in the data, and it is not a model that outputs a finished DXF by itself.

## Technical details

**The data**
- ResPlan: 17,000 residential floor plans (vector geometry). After cleaning (augmented copies, twins, broken floors, quality problems removed): **15,832 plans**.
- Split: **12,682 train · 1,580 validation · 1,570 test**.
- Each plan becomes a **room map**: 8 layers of 128 × 128 pixels (living, bedroom, bathroom, kitchen, balcony, wall, door, window). One pixel ≈ 13 cm (median). Drawing all the maps took about 4 minutes.
- Bedrooms in the data: 1 → 1,297 plans · 2 → 6,005 · 3 → 4,572 · 4 → 616 · 5+ → 192. 77% of the plans have a balcony.
- A room table from 3,000 plans (typical room sizes, which room opens to which) is used by the rules in step 3.

**The model: a conditional VAE**
- Input of the decoder: the silhouette (128 × 128) + a condition of 7 numbers (bedrooms 1–5+ as one-hot, balcony 0/1, log of the area) + a code of **32 random numbers**.
- The decoder sees the silhouette at 5 sizes (8, 16, 32, 64, 128 px), so the rooms stay inside the outline.
- **1,518,448 parameters** (6 MB in PyTorch, 3 MB as ONNX).
- Loss: binary cross-entropy + Dice (room shapes) + β · KL (β = 0.008, free bits 0.1 per code number).
- Adam, learning rate 0.001, batch 128.

**The training**
- On a laptop **CPU** (no GPU), PyTorch.
- First, 3 short trials to choose β (4 epochs on 40% of the data each): about **11 minutes**.
- Then the full training: up to 40 epochs, stopped when the score did not improve for 8 epochs. It stopped after **28 epochs**, best at epoch 20: **68 minutes**, about 2 minutes per epoch.

**How well it works (validation plans the model never saw)**
- The balcony is as asked in **99%** of the pictures.
- The number of bedrooms is right in **37%** of the raw pictures, ±1 in **86%** (step 3 fixes or drops the rest).
- Two pictures for the same request differ in **58%** of their pixels: the model really gives different options.
- One request on a laptop (30 pictures + step 3 + 5 vector plans): about **1.5 seconds**.

## Where each step is

| Step | What | Code |
|---|---|---|
| Training | room maps from ResPlan, conditional VAE | `01_autoencoder/notebook/04_layout_generator.ipynb` |
| Step 2 + 3 | the model's pictures → clean vector plans | `01_autoencoder/notebook/05_layout_to_vector.ipynb` (figure 50 shows the raw model output next to the final plan) |
| Step 4 | photo → silhouette, and the whole app end to end | `01_autoencoder/notebook/06_photo_to_silhouette.ipynb` |
| All steps | shared code | `01_autoencoder/notebook/layout_core.py` |
| The app | model server (API + test page) | `02_server/` |
