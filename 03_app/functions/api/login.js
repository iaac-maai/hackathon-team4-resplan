// POST /api/login from the login page: form fields email, password, mode ("login" or "signup").
// Supabase checks the password (or creates the account) and sends back a session; we put it in cookies.
import { supabase, sessionCookies } from "../_middleware.js";

export async function onRequestPost({ request, env }) {
  const url = new URL(request.url);
  const form = await request.formData();
  const email = String(form.get("email") || "").trim().toLowerCase();
  const password = String(form.get("password") || "");
  const mode = form.get("mode") === "signup" ? "signup" : "login";
  const back = (code) => Response.redirect(new URL(`/login?mode=${mode}&error=${code}&email=${encodeURIComponent(email)}`, url), 302);

  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return back("email");
  if (password.length < 8) return back("short");

  const session = mode === "signup"
    ? await supabase(env, "/auth/v1/signup", { email, password })
    : await supabase(env, "/auth/v1/token?grant_type=password", { email, password });

  if (!session || !session.access_token) {
    const e = String(session && session.error || "");
    if (mode === "signup" && /exists|registered/.test(e + (session && session.message))) return back("exists");
    if (mode === "signup" && session && session.id) return back("confirm");   // "Confirm email" is still on in Supabase
    if (/rate|over_/.test(e)) return back("busy");
    return back(mode === "signup" ? "signup" : "wrong");
  }

  const res = new Response(null, { status: 302, headers: { Location: "/" } });
  for (const c of sessionCookies(url, session)) res.headers.append("Set-Cookie", c);
  return res;
}
