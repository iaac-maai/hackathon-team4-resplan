# 1 · System diagram: Sketcheat (Team 4)

Every box and arrow below exists in our code (`03_app/` and the Render server).

```mermaid
flowchart LR
  U["User's browser<br/>index.html"] -->|"HTTPS"| A["Cloudflare Access<br/>(email PIN, optional)"]
  A --> M["Pages middleware<br/>functions/_middleware.js<br/>team password cookie"]
  M --> F["Pages Functions<br/>functions/api/*"]
  F -->|"POST /api/design (JSON, photo as base64)"| R["Render: team4-layout-server<br/>LayoutCVAE decoder (ONNX)"]
  F -->|"GET /health"| R
  F -->|"INSERT / SELECT"| D[("Cloudflare D1<br/>hackathon-example-db<br/>designs · saved_layouts")]
```

| Arrow | Code |
|---|---|
| Browser → Functions | `fetch("/api/design")`, `/api/wake`, `/api/save`, `/api/saved`, `/api/dxf/<id>` in `public/index.html` |
| Login | `functions/api/login.js` → Supabase `/auth/v1/signup` or `/auth/v1/token`, sets HttpOnly cookies `sb_access` + `sb_refresh` |
| Who is this? | `functions/_middleware.js` → Supabase `/auth/v1/user` on every request, puts the user id in `data.user` |
| Functions → Render | `fetch(env.MODEL_URL + "/api/design")` in `functions/api/design.js`, `/health` in `functions/api/wake.js` |
| Functions → D1 | `env.DB.prepare(...)` in `design.js` (table `designs`), `save.js`, `saved.js`, `dxf/[id].js` (table `saved_layouts`). Every query has `WHERE user_id = ?` with the id from the session |

No R2: the sketch thumbnail (200 px JPEG) is stored as text in D1.
