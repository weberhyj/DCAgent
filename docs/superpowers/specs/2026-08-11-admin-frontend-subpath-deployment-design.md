# Admin frontend configurable subpath deployment

## Goal

Allow the Vue administration frontend to be built and served from a configurable URL subpath,
with `/admin/` as the production default, while keeping backend API requests rooted at `/api`.

Example production URLs:

- Administration UI: `http://intranet-host/admin/`
- Administration route: `http://intranet-host/admin/knowledge`
- Backend API: `http://intranet-host/api/knowledge/sources`

## Current state

The administration frontend currently assumes it is hosted at `/`:

- Vite does not configure `base`.
- Vue Router calls `createWebHistory()` without a history base.
- The favicon and administration brand image use `/favicon-logo.svg`.
- The Axios client intentionally uses `/api` as its root-relative base URL.
- The repository does not contain an administration-frontend reverse-proxy subpath example.

Hosting the current build below `/admin/` would therefore break static assets and direct navigation
or refreshes of history-mode routes.

## Selected approach

Use a build-time `VITE_ADMIN_BASE_PATH` variable as the single source of truth for the administration
frontend public base. The default is `/admin/`.

The value is normalized so it:

- starts with `/`;
- ends with `/`;
- rejects query strings, fragments, absolute URLs, and ambiguous traversal segments;
- supports `/` when an operator explicitly needs a root deployment.

Vite uses the normalized value as its `base`. Vue Router uses Vite's generated
`import.meta.env.BASE_URL` as the `createWebHistory` base so router and asset behavior cannot drift.

The Axios client remains root-relative at `/api`. The administration base path must never be
prepended to backend requests.

## Components and responsibilities

### Base-path configuration

A small configuration helper normalizes and validates `VITE_ADMIN_BASE_PATH`. Vite imports this
helper when creating its configuration. Runtime frontend code consumes `import.meta.env.BASE_URL`,
which is the normalized value emitted by Vite.

This keeps configuration parsing isolated from components and prevents multiple implementations of
path normalization.

### Vite build

`admin-frontend/vite.config.ts` will load the environment for the active mode, obtain
`VITE_ADMIN_BASE_PATH`, normalize it, and assign it to Vite's `base` option.

Development and production use the same path behavior. An operator may override the default:

```bash
VITE_ADMIN_BASE_PATH=/operations/ pnpm --filter dc-agent-admin-frontend build
```

### Vue Router

`admin-frontend/src/router/index.ts` will create history with:

```ts
createWebHistory(import.meta.env.BASE_URL)
```

Application route definitions stay root-relative inside the router. For example, the internal
`/knowledge` route becomes `/admin/knowledge` externally when the configured base is `/admin/`.

Route-name navigation and existing route-path checks continue to operate on application paths and
do not hard-code the deployment prefix.

### Static assets

The HTML favicon will use Vite's HTML base placeholder. Component-level public assets, including the
administration brand image, will resolve relative to `import.meta.env.BASE_URL` instead of `/`.

Bundled assets imported through TypeScript or Vue SFCs remain managed by Vite and require no special
handling.

### Backend API

`admin-frontend/src/services/api.ts` keeps:

```ts
baseURL: '/api'
```

This is deliberately independent of `VITE_ADMIN_BASE_PATH`. The reverse proxy continues to expose
the backend at `/api/`.

## Reverse-proxy contract

For a `/admin/` build, Nginx serves the generated `admin-frontend/dist` directory beneath that URL
and falls back to the administration `index.html` for history-mode routes:

```nginx
location = /admin {
    return 301 /admin/;
}

location /admin/ {
    alias /opt/dc-agent/admin-frontend/dist/;
    try_files $uri $uri/ /admin/index.html;
}

location /api/ {
    proxy_pass http://127.0.0.1:8000/api/;
}
```

The configured frontend base and the Nginx location must match exactly. Changing the deployment
subpath requires a rebuild because the base is embedded into generated asset URLs.

## Error handling

The build fails with a clear configuration error when `VITE_ADMIN_BASE_PATH` is malformed. This is
preferable to producing a build whose asset and router URLs disagree.

The reverse proxy redirects `/admin` to `/admin/` so relative URL resolution and router initialization
remain deterministic.

## Testing and verification

Automated coverage will verify:

- default normalization to `/admin/`;
- custom normalization such as `/operations/`;
- explicit root deployment `/`;
- rejection of absolute URLs, query strings, fragments, and traversal segments;
- Vue Router uses `import.meta.env.BASE_URL`;
- the favicon and brand image no longer resolve from `/favicon-logo.svg`;
- Axios continues to use `/api`;
- a production build generated with `/admin/` references subpath-prefixed assets.

Deployment verification will include:

1. Build with `VITE_ADMIN_BASE_PATH=/admin/` using pnpm.
2. Serve the generated directory with the documented Nginx configuration.
3. Open `/admin/`, `/admin/knowledge`, and a nested detail route directly.
4. Refresh each route and confirm Nginx returns the SPA entry document.
5. Confirm API calls go to `/api/...`, not `/admin/api/...`.

## Scope

In scope:

- Administration frontend Vite base configuration.
- Vue Router history base.
- Public favicon and brand image paths.
- Tests and an Nginx deployment example.

Out of scope:

- Moving backend endpoints below the administration subpath.
- Changing the user-facing `frontend` application.
- Adding authentication or authorization.
- Runtime base-path switching without rebuilding the administration frontend.
