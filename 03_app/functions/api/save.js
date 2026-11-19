// POST /api/save with JSON: { request_id, rank, area_m2, bedrooms, balcony, option }
// Called when the user swipes a layout to the right: one row in the D1 table saved_layouts
// (with the DXF file as text, about 40 KB, so no R2 is needed). Sends the new id back.
const KEEP = ["rooms", "walls", "doors", "windows", "rails", "entrance", "outline", "doors_info", "score_parts"];
const ROOM_TYPES = new Set(["living", "bedroom", "bathroom", "kitchen", "balcony"]);
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

// The table for saved layouts makes itself the first time it is needed:
// no extra database command after a deploy, and the designs table is never touched.
export async function ensureTable(env) {
  await env.DB.prepare(
    "CREATE TABLE IF NOT EXISTS saved_layouts (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, request_id TEXT NOT NULL, " +
    "area_m2 REAL NOT NULL, bedrooms INTEGER NOT NULL, balcony INTEGER NOT NULL, option_rank INTEGER NOT NULL, score REAL, " +
    "summary TEXT, plan TEXT NOT NULL, dxf_base64 TEXT, user_id TEXT)",
  ).run();
}

export async function onRequestPost({ request, env, data }) {
  await ensureTable(env);
  if (Number(request.headers.get("Content-Length") || 0) > 2 * 1024 * 1024) return bad("Too large.", 413);
  let body;
  try { body = await request.json(); } catch { return bad("Send JSON."); }

  const { request_id: requestId, option } = body;
  const rank = parseInt(body.rank, 10);
  const area = Number(body.area_m2);
  const bedrooms = parseInt(body.bedrooms, 10);
  const balcony = Number(body.balcony) === 1 ? 1 : 0;
  if (!UUID.test(String(requestId))) return bad("Unknown request.");
  if (!option || !Array.isArray(option.rooms)) return bad("No layout to save.");
  if (!(rank >= 1 && rank <= 20) || !(area >= 25 && area <= 400) || !(bedrooms >= 1 && bedrooms <= 5)) return bad("Wrong values.");

  // the request must exist AND belong to this user (design.js wrote it to the designs table)
  const known = await env.DB.prepare("SELECT 1 FROM designs WHERE id = ? AND user_id = ?").bind(requestId, data.user.id).first();
  if (!known) return bad("Unknown request.");

  // only the parts of the plan the page needs to draw it again; room types from a fixed list
  const plan = {};
  for (const k of KEEP) if (option[k] !== undefined) plan[k] = option[k];
  plan.rooms = option.rooms
    .filter((r) => ROOM_TYPES.has(r.type))
    .map((r) => ({ type: r.type, area_m2: Number(r.area_m2) || 0, polygons: r.polygons }));

  const dxf = typeof option.dxf_base64 === "string" && /^[A-Za-z0-9+/=]+$/.test(option.dxf_base64)
    && option.dxf_base64.length < 500000 ? option.dxf_base64 : null;

  // one row in D1. The "?" placeholders keep user input out of the SQL text
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  await env.DB.prepare(
    "INSERT INTO saved_layouts (id, created_at, request_id, area_m2, bedrooms, balcony, option_rank, score, summary, plan, dxf_base64, user_id) " +
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(id, createdAt, requestId, area, bedrooms, balcony, rank,
          Number.isFinite(Number(option.score)) ? Number(option.score) : null,
          JSON.stringify(option.summary ?? {}), JSON.stringify(plan), dxf, data.user.id)
    .run();

  return Response.json({ ok: true, id, created_at: createdAt });
}

function bad(message, status = 400) {
  return Response.json({ ok: false, error: message }, { status });
}
