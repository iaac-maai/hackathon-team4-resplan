// POST /api/design with form fields: photo (file), area_m2, bedrooms, balcony, seed
// 1. ask the layout server for 5 options, 2. store the sketch in R2,
// 3. store one row in D1 (with option 1's drawing as JSON), 4. send the options back to the page.
const WANT_DXF = 1; // 0 = no DXF files (smaller, faster answer)

export async function onRequestPost({ request, env }) {
  const form = await request.formData();
  const photo = form.get("photo");
  const area = Number(form.get("area_m2"));
  const bedrooms = parseInt(form.get("bedrooms"), 10);
  const balcony = form.get("balcony") === "1" ? 1 : 0;
  const seed = parseInt(form.get("seed") || "0", 10) || 0;

  if (!photo || typeof photo === "string") return bad("Choose a photo of the outline.");
  if (photo.size > 8 * 1024 * 1024) return bad("Photo too large (max 8 MB).", 413);
  if (!(area >= 25 && area <= 400)) return bad("The area must be between 25 and 400 m².");
  if (!(bedrooms >= 1 && bedrooms <= 5)) return bad("Bedrooms must be between 1 and 5.");

  const bytes = new Uint8Array(await photo.arrayBuffer());

  // 1. Ask the layout server (Render). If it is asleep, it can take about a minute to answer.
  //    preview: 0 -> the page draws the plans itself from the vectors (sharper, and a smaller answer)
  let answer;
  try {
    const res = await fetchWithTimeout(`${env.MODEL_URL}/api/design`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photo_base64: toBase64(bytes), area_m2: area, bedrooms, balcony, seed,
                             dxf: WANT_DXF, preview: 0 }),
    }, 180000);
    answer = await res.json();
  } catch (err) {
    return Response.json({ ok: false, waking: true,
      error: "The layout model is waking up (the free server sleeps after 15 minutes). Please try again in about a minute.",
      detail: String(err) }, { status: 503 });
  }
  if (!answer.ok) return Response.json(answer); // e.g. no closed shape in the photo: nothing to save

  // 2. Store the sketch in R2
  const id = crypto.randomUUID();
  const photoKey = `uploads/${id}`;
  await env.BUCKET.put(photoKey, bytes, { httpMetadata: { contentType: photo.type || "image/jpeg" } });

  // 3. One row in D1. The "?" placeholders keep user input out of the SQL text
  const best = answer.options[0];
  const bestPlan = { rooms: best.rooms, walls: best.walls, doors: best.doors, windows: best.windows,
                     rails: best.rails, entrance: best.entrance, outline: best.outline };
  const createdAt = new Date().toISOString();
  await env.DB.prepare(
    "INSERT INTO designs (id, created_at, photo_key, area_m2, bedrooms, balcony, n_options, best_score, summary, best_plan) " +
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(id, createdAt, photoKey, area, bedrooms, balcony, answer.options.length,
          best.score ?? null, JSON.stringify(best.summary ?? {}), JSON.stringify(bestPlan))
    .run();

  // 4. Answer the page
  return Response.json({ id, created_at: createdAt, ...answer });
}

function bad(message, status = 400) {
  return Response.json({ ok: false, error: message }, { status });
}

async function fetchWithTimeout(url, options = {}, ms = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function toBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}
