// GET /api/logout: end the session at Supabase and remove the cookies
import { supabase, readCookies, clearCookies } from "../_middleware.js";

export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const { sb_access } = readCookies(request.headers.get("Cookie"));
  if (sb_access) await supabase(env, "/auth/v1/logout", {}, sb_access);
  const res = new Response(null, { status: 302, headers: { Location: "/login?bye=1" } });
  for (const c of clearCookies(url)) res.headers.append("Set-Cookie", c);
  return res;
}
