# Sketcheat · Final hand-in

## Use case

Sketcheat is an early-stage layout exploration app for architects. A user signs up or logs in with an email and password, uploads or takes a photo of a flat outline, then enters the area, bedroom count and balcony preference. The app requests generated layouts from the layout server, displays their plans and scores, and lets the user save a layout, edit rooms and download a DXF.

## Live links

| Part | URL |
|---|---|
| App | [sketcheat-team4.pages.dev](https://sketcheat-team4.pages.dev) |
| Layout model server and health | [team4-layout-server.onrender.com](https://team4-layout-server.onrender.com) · [/health](https://team4-layout-server.onrender.com/health) |
| Search server | [team4-resplan-server.onrender.com](https://team4-resplan-server.onrender.com) |
| Hugging Face model (Monday autoencoder) | [iaac-maai/hackathon-team4-resplan-ae](https://huggingface.co/iaac-maai/hackathon-team4-resplan-ae) |
| Privacy page | [sketcheat-team4.pages.dev/privacy.html](https://sketcheat-team4.pages.dev/privacy.html) |

The linked Hugging Face repository is for Monday's autoencoder, not the layout model used by Sketcheat. The layout server currently loads its ONNX artifact from `02_server/model/`; publishing that artifact to Hugging Face remains pending.

## How visitors get in

Create an account or log in from the app using an email address and password. Supabase Auth manages the account; Pages middleware checks the session and supplies the user id to protected API functions. There is no shared team password and no Cloudflare Access gate. Course test-account details are distributed separately and are not stored in this repository.

## Diagrams

- [System](diagrams/1_system.md)
- [User flow](diagrams/2_user_flow.md)
- [Generate-and-save sequence](diagrams/3_sequence.md)
- [Wednesday user stories and three app sequences](../03_app/diagrams/README.md)

## What is clear

The app can turn a photographed outline and a few requirements into ranked layout candidates, then save an option as a vector plan. The model generates possibilities; the rules and ranking make some candidates usable, but do not replace architectural judgement.

## What is still unclear

We have not measured quality across a representative held-out set of unusual outlines. A high score is not evidence that a plan is comfortable, accessible or compliant with local building rules.

## What we would do next

We would publish and version the exact deployed layout model on Hugging Face, then evaluate it on a documented held-out set with architects reviewing both successful layouts and failures.

## Limitations

- The model can generate awkward room relationships or the wrong bedroom count.
- Rule-based cleanup can reject most generated maps; the recorded server example returned four options instead of five after generating 60 maps.
- The output score is a ranking heuristic, not a measure of architectural compliance.
- The app has no guarantee that generated plans satisfy local codes, accessibility requirements or a user's full functional needs.
- Render's free service sleeps when idle, so the first request can take about a minute; some requests can return fewer than five candidates.
- The layout model is currently stored in the repository and loaded from the Render service's local model directory. It has not yet been published to Hugging Face.

See [`PROOFS.md`](PROOFS.md) for database, authentication and service evidence.
