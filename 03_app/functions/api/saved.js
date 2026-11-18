// GET /api/saved: the saved layouts (swiped right), newest first, read from D1.
// The DXF text is left out here (it is large); has_dxf says whether /api/dxf/<id> has one.
import { ensureTable } from "./save.js";

export async function onRequestGet({ env }) {
  await ensureTable(env);
  const { results } = await env.DB.prepare(
    "SELECT id, created_at, request_id, area_m2, bedrooms, balcony, option_rank, score, summary, plan, " +
    "dxf_base64 IS NOT NULL AS has_dxf FROM saved_layouts ORDER BY created_at DESC LIMIT 60",
  ).all();
  return Response.json(results);
}

// DELETE /api/saved?id=<id>: remove one saved layout from the gallery
export async function onRequestDelete({ request, env }) {
  const id = new URL(request.url).searchParams.get("id") || "";
  if (!/^[0-9a-f-]{36}$/.test(id)) return Response.json({ ok: false, error: "Unknown layout." }, { status: 400 });
  await ensureTable(env);
  await env.DB.prepare("DELETE FROM saved_layouts WHERE id = ?").bind(id).run();
  return Response.json({ ok: true });
}
