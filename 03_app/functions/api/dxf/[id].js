// GET /api/dxf/<id>: the DXF file of a saved layout, read back from D1 (behind the password like everything else)
import { ensureTable } from "../save.js";

export async function onRequestGet({ params, env }) {
  const id = String(params.id || "");
  if (!/^[0-9a-f-]{36}$/.test(id)) return new Response("Not found", { status: 404 });
  await ensureTable(env);
  const row = await env.DB.prepare("SELECT dxf_base64 FROM saved_layouts WHERE id = ?").bind(id).first();
  if (!row || !row.dxf_base64) return new Response("Not found", { status: 404 });
  const bytes = Uint8Array.from(atob(row.dxf_base64), (c) => c.charCodeAt(0));
  return new Response(bytes, {
    headers: { "Content-Type": "application/dxf", "Content-Disposition": `attachment; filename="sketcheat_${id.slice(0, 8)}.dxf"` },
  });
}
