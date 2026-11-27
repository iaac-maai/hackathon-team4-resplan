# Wednesday app diagrams

These diagrams describe the app that was built, not an aspirational Cloudflare Access/team-password design. The app uses Supabase Auth and individual email/password accounts.

## User stories

1. **Generate options:** As an architect exploring an early design, I want to photograph a flat outline and specify area, bedrooms and balcony, so I can compare generated layout options.
2. **Keep and export a design:** As a logged-in user, I want to save a candidate to my gallery, edit its rooms and download its DXF, so I can revisit and continue working with it.

## System

```mermaid
flowchart LR
  B["Browser<br/>Sketcheat UI"] -->|"HTTPS"| P["Cloudflare Pages"]
  P --> F["Pages Functions<br/>middleware + API"]
  F -->|"login / refresh / identify user"| S["Supabase Auth"]
  F -->|"design + health requests"| R["Render<br/>team4-layout-server<br/>ONNX + geometry rules"]
  F -->|"user-scoped reads/writes"| D[("Cloudflare D1<br/>designs · saved_layouts")]
  G["GitHub deployment<br/>02_server/model/layout_decoder.onnx"] --> R
  H["Hugging Face<br/>layout artifact publication pending"] -.->|"planned model hosting"| R
```

The Hugging Face arrow is a planned artifact-hosting path, not a live dependency. The Render service currently loads the checked-in ONNX file. There is no team-password cookie, Cloudflare Access gate or R2 bucket.

## Sequences

### 1. Sign up or log in

```mermaid
sequenceDiagram
  participant B as Browser
  participant F as Pages Functions
  participant M as Middleware
  participant S as Supabase Auth
  B->>F: POST /api/login (email, password, mode)
  F->>S: /auth/v1/signup or /auth/v1/token
  S-->>F: Auth session
  F-->>B: Set HttpOnly session cookies; redirect
  B->>M: Request protected page or API
  M->>S: GET /auth/v1/user (access cookie)
  S-->>M: User id and email
  M-->>F: Continue with context.data.user
```

### 2. Wake the server and generate layouts

```mermaid
sequenceDiagram
  participant B as Browser
  participant F as Pages Functions
  participant R as Render layout server
  participant D as D1
  B->>F: GET /api/wake
  F->>R: GET /health
  R-->>F: Ready, or unavailable while waking
  F-->>B: Health status
  B->>F: POST /api/design (photo, area, bedrooms, balcony)
  F->>R: POST /api/design
  alt Render is sleeping
    R-->>F: 503 {waking: true}
    F-->>B: Waking response
    B->>F: Retry request
    F->>R: POST /api/design
  end
  R-->>F: Ranked options, scores, previews and DXFs
  F->>D: INSERT design with authenticated user_id
  F-->>B: Options
```

### 3. Save, view and download

```mermaid
sequenceDiagram
  participant B as Browser
  participant F as Pages Functions
  participant D as D1
  B->>F: POST /api/save (design and chosen option)
  F->>D: INSERT saved_layouts with context.data.user.id
  D-->>F: Saved row id
  F-->>B: Saved confirmation
  B->>F: GET /api/saved
  F->>D: SELECT rows WHERE user_id = authenticated id
  D-->>F: This user's gallery
  F-->>B: Gallery layouts
  B->>F: GET /api/dxf/{id}
  F->>D: SELECT DXF WHERE id = ? AND user_id = authenticated id
  D-->>F: This user's DXF only
  F-->>B: DXF download
```
