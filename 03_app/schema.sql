-- One row per request to the layout generator.
-- On your laptop: npm run db:local     Online (only the FIRST time): npm run db:remote
-- WARNING: the DROP lines delete the saved designs. Run this file again only to start fresh.
DROP TABLE IF EXISTS predictions;            -- the template's old table
DROP TABLE IF EXISTS designs;
CREATE TABLE designs (
  id TEXT PRIMARY KEY,        -- a random id (crypto.randomUUID)
  created_at TEXT NOT NULL,   -- ISO date and time, UTC
  photo_thumb TEXT,           -- a small thumbnail of the sketch (JPEG data URL, about 10 KB)
  area_m2 REAL NOT NULL,      -- what the user asked
  bedrooms INTEGER NOT NULL,
  balcony INTEGER NOT NULL,   -- 1 = yes, 0 = no (SQLite has no boolean)
  n_options INTEGER NOT NULL, -- how many options came back (1-5)
  best_score REAL,            -- score of option 1 (0-1)
  summary TEXT,               -- option 1 in short, as JSON (bedrooms, bathrooms, m²)
  best_plan TEXT              -- option 1's drawing (rooms, walls, doors, windows) as JSON, in metres
);
CREATE INDEX idx_designs_created_at ON designs (created_at);

-- One row per layout the user SAVED (swiped to the right).
-- The app also makes this table by itself the first time a layout is saved (functions/api/save.js).
CREATE TABLE IF NOT EXISTS saved_layouts (
  id TEXT PRIMARY KEY,          -- a random id (crypto.randomUUID)
  created_at TEXT NOT NULL,     -- ISO date and time, UTC
  request_id TEXT NOT NULL,     -- the row in designs it came from
  area_m2 REAL NOT NULL,        -- what the user asked
  bedrooms INTEGER NOT NULL,
  balcony INTEGER NOT NULL,     -- 1 = yes, 0 = no
  option_rank INTEGER NOT NULL, -- which of the options it was (1 = best score)
  score REAL,                   -- the model's score for this layout (0-1)
  summary TEXT,                 -- JSON: bedrooms, bathrooms, m²
  plan TEXT NOT NULL,           -- JSON: rooms, walls, doors, windows, outline, ... in metres
  dxf_base64 TEXT               -- the DXF file (about 40 KB), so it can be downloaded later
);
CREATE INDEX IF NOT EXISTS idx_saved_created_at ON saved_layouts (created_at);
