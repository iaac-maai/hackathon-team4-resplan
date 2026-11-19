// GET /api/wake: is the layout server awake? Asking also wakes it up.
export async function onRequestGet({ env }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(`${env.MODEL_URL}/health`, { signal: controller.signal });
    return Response.json({ ready: res.ok });
  } catch {
    return Response.json({ ready: false });
  } finally {
    clearTimeout(timer);
  }
}
