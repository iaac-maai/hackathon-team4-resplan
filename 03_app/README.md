# Sketcheat

**Public app:** https://sketcheat-team4.pages.dev

Sketcheat helps an architect explore early flat layouts. Sign up or log in with an email address and password; accounts are managed by Supabase Auth. The user uploads or takes a photo of a flat outline, enters the area, bedroom count and balcony preference, and gets a deck of generated layouts. A user can save a layout to their gallery, edit its rooms, and download its DXF.

The app's supporting public services are:

| Service | Use |
|---|---|
| [Layout model server](https://team4-layout-server.onrender.com) | Render runs the ONNX layout decoder and geometry pipeline; `/health` reports server readiness. |
| [Search server](https://team4-resplan-server.onrender.com) | Existing-plan search endpoint, maintained separately from layout generation. |
| [Privacy page](https://sketcheat-team4.pages.dev/privacy.html) | Public explanation of app data handling. |

## Cloudflare and connected services

| Service | Why it is used |
|---|---|
| Cloudflare Pages | Serves the static app and routes requests to Pages Functions. |
| Cloudflare Pages Functions | Implements the login gate and API, calls Supabase and the Render layout server, and enforces the logged-in user identity. |
| Cloudflare D1 | Stores generated-design metadata, saved layouts, thumbnails and DXF data, scoped to the logged-in user's id. |
| Supabase Auth (external) | Creates accounts, verifies email/password credentials, and supplies the user identity used by the middleware. |
| Render (external) | Hosts the separate ONNX layout server. The first request after its free service sleeps can take about a minute. |

The app does not use R2: the small sketch thumbnail and saved DXF data are stored in D1. A visitor creates an account or logs in through the app; there is no shared team-password cookie and no Cloudflare Access login gate.

## Wednesday diagrams compared with the built app

The Wednesday design placed the browser, authentication, API, model server and database in a single request flow; the built app follows that flow, with Supabase Auth providing the identity checked by the Pages middleware.
The live app uses Pages Functions to proxy model requests and D1 for user-owned designs, saved layouts and DXFs; it does not use R2.
The initial system sketch's team-password cookie was not implemented: access is by an individual Supabase email/password account.
The production system also handles a sleeping Render server by returning a waking response and retrying, and restricts every saved-layout query to the authenticated user's id.

## Diagrams

The Wednesday user stories, system diagram and three sequence diagrams are in [`diagrams/`](diagrams/). The final review diagrams and evidence are in [`../04_final/`](../04_final/).
