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
