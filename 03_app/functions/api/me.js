// GET /api/me: who is logged in (for the header of the page)
export async function onRequestGet({ data }) {
  return Response.json({ email: data.user.email });
}
