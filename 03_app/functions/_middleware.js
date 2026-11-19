// Runs before EVERY request: the login gate.
// Users sign up and log in with email + password. Supabase Auth keeps the accounts and hashes the
// passwords on its own servers (so no heavy hashing runs here, and no Error 1102 on the free plan).
// The browser gets two HttpOnly cookies: sb_access (1 hour) and sb_refresh (30 days).
// Every request: cookie -> ask Supabase "who is this?" -> user id -> context.data.user for the API.

const PUBLIC_PATHS = ["/login", "/login.html", "/api/login", "/api/logout", "/style.css", "/privacy", "/privacy.html"];

export async function onRequest(context) {
  const { request, env, next, data } = context;
  const url = new URL(request.url);
  if (PUBLIC_PATHS.includes(url.pathname)) return next();

  const cookies = readCookies(request.headers.get("Cookie"));
  let user = cookies.sb_access ? await getUser(env, cookies.sb_access) : null;
  let fresh = null;
  if (!user && cookies.sb_refresh) {            // the 1-hour access token ran out: get a new one
    fresh = await supabase(env, "/auth/v1/token?grant_type=refresh_token", { refresh_token: cookies.sb_refresh });
    if (fresh && fresh.access_token) user = fresh.user; else fresh = null;
  }

  if (!user) {
    if (url.pathname.startsWith("/api/")) return Response.json({ error: "Not logged in" }, { status: 401 });
    return Response.redirect(new URL("/login", url), 302);
  }

  data.user = { id: user.id, email: user.email };   // the API functions use ONLY this id, never one from the browser
  const res = await next();
  if (!fresh) return res;
  const out = new Response(res.body, res);
  for (const c of sessionCookies(url, fresh)) out.headers.append("Set-Cookie", c);
  return out;
}

// ---------- small helpers, also used by api/login.js and api/logout.js ----------
export async function supabase(env, path, body, token) {
  try {
    const res = await fetch(env.SUPABASE_URL + path, {
      method: body ? "POST" : "GET",
      headers: { apikey: env.SUPABASE_KEY, "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json().catch(() => ({}));
    return res.ok ? json : { error: json.error_code || json.code || json.error || "error", message: json.msg || json.error_description || json.message || "" };
  } catch (err) {
    return { error: "unreachable", message: String(err) };
  }
}

async function getUser(env, token) {
  const u = await supabase(env, "/auth/v1/user", null, token);
  return u && u.id ? u : null;
}

export function readCookies(header) {
  const out = {};
  for (const part of (header || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

// HttpOnly: page scripts can't read them. SameSite=Lax: not sent by other sites' forms. Secure online.
export function sessionCookies(url, session) {
  const secure = url.protocol === "https:" ? "; Secure" : "";
  const base = `; Path=/; HttpOnly; SameSite=Lax${secure}`;
  return [
    `sb_access=${encodeURIComponent(session.access_token)}; Max-Age=${session.expires_in || 3600}${base}`,
    `sb_refresh=${encodeURIComponent(session.refresh_token)}; Max-Age=${60 * 60 * 24 * 30}${base}`,
  ];
}

export function clearCookies(url) {
  const secure = url.protocol === "https:" ? "; Secure" : "";
  return ["sb_access", "sb_refresh", "auth"].map((n) => `${n}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax${secure}`);
}
