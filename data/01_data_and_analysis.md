# 1 · Data & analysis

Built in `01_autoencoder/notebook/03_make_tensors.ipynb` from `ResPlan_clean.pkl`.
Large files are NOT in Git; get them from the team Google Drive (link: https://drive.google.com/drive/folders/1oXMSzjlTJBT9qtYLofzUk8Z7TvTkukLF?usp=drive_link).

**Row i is the same plan in both .npz files. Use `ids[i]` to match `split.json` and the PNGs.**

## resplan_tensors.npz
| Key | Shape | Meaning |
|---|---|---|
| images | (17000, 64, 64) uint8 | 0 empty, 1 living, 2 bedroom, 3 kitchen, 4 bathroom, 5 balcony, 6 storage, 7 stair, 8 wall, 9 door, 10 window, 11 front_door |
| ids | (17000,) | ResPlan plan id |
| split | (17000,) | 0 train, 1 val, 2 test, 3 augmented (official split.json) |
| area_m2, area_sqft | (17000,) | real floor area (NaN for 6 plans with a bad area) |
| bedrooms, bathrooms | (17000,) | counts |
| room_counts | (17000, 8) | count per type: living, bedroom, kitchen, bathroom, balcony, storage, stair, front_door |
| relation | (17000, 8, 8) | number of connections between room types (same order) |
| adjacency | (17000, 23, 23) | room-to-room: 0 none, 1 door, 2 adjacency (open or shared wall), 3 window, 4 front door |
| node_type | (17000, 23) | type index of each room slot, -1 = empty slot |
| node_area | (17000, 23) | room's share of total room area |
| room_box | (17000, 23, 4) | centre x, centre y, width, height, 0–1 (y points up, as in the drawing) |
| tags, tag_names | (17000, 8) | cleaning tags from Day 1 |

## plan_features.npz
| Key | Shape | Meaning |
|---|---|---|
| depth | (17000, 23) | steps from front door per room (-1 = not reachable) |
| mean_depth, max_depth | (17000,) | how deep the home is |
| circulation_score | (17000,) | share of rooms within 2 steps of the entrance (0–1, higher = more compact) |
| bedroom_passthrough | (17000,) | bedrooms reached through another bedroom/bathroom |
| missing_door | (17000,) | a room can't be reached through doors (13% of plans) |
| window_ratio | (17000, 23) | window area ÷ floor area per room (≥ 0.10 = good daylight) |
| window_sides | (17000, 23) | bitmask: 1 right, 2 top, 4 left, 8 bottom (of the drawing) |
| balcony_lit | (17000, 23) | room has no window but opens onto a balcony |
| dark_rooms | (17000,) | living/bedroom/kitchen with no window and no balcony |
| daylight_score | (17000,) | share of living/bedroom/kitchen with daylight (0–1) |
| cross_vent_rooms | (17000,) | rooms with windows on opposite walls |

## plans_png/<id>.png
256 × 256 picture of each plan, white background, same colours as the notebook grids.

## Example: search
~~~python
import numpy as np
d = np.load("resplan_tensors.npz"); f = np.load("plan_features.npz")
m = (d["bedrooms"] == 3) & (abs(d["area_sqft"] - 1100) <= 110)
ids = d["ids"][m]    # -> plans_png/<id>.png, scores in f[...][m]
~~~

## Caveats
- No north direction in the data: "sides" are sides of the drawing, not orientation.
- All plans are South Asian flats. Data licence: credit ResPlan, non-commercial use only.