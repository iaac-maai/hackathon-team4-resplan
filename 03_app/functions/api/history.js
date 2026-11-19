// GET /api/history: the 12 most recent designs, read from D1
export async function onRequestGet({ env }) {
  const { results } = await env.DB.prepare(
    "SELECT id, created_at, photo_thumb, area_m2, bedrooms, balcony, n_options, best_score, summary, best_plan " +
    "FROM designs ORDER BY created_at DESC LIMIT 12",
  ).all();
  return Response.json(results);
}
