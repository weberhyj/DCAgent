# Unified Page Logo Design

## Goal

Use the supplied `favicon-logo.svg` as the brand mark across the user application, the administration application, and both browser-tab favicons without changing the existing product names or page layouts.

## Current State

The user application displays a text-only `DC` mark in `CompanyBrand.vue`. The administration application displays a separate text-only `DC` mark in `AdminLayout.vue`. Neither Vite application currently declares a favicon in its `index.html`.

The supplied SVG is a square `66.77 × 66.77` Huawei logo with a transparent outer canvas, a red gradient symbol, and black `HUAWEI` lettering.

## Decision

Each Vite application owns a public copy of the same source SVG:

- `frontend/public/favicon-logo.svg`
- `admin-frontend/public/favicon-logo.svg`

This deliberate duplication keeps the two applications independently buildable and deployable. A shared path outside either Vite root or a filesystem link would add build and deployment coupling for a single small asset.

Both visible brand marks use a normal `<img src="/favicon-logo.svg">`. The existing `DC-Agent` text remains unchanged, so the image is decorative and uses an empty alternative text plus `aria-hidden="true"`. No SVG markup is injected into the DOM, and no `v-html` is introduced.

## User Application

In `frontend/src/components/chat/CompanyBrand.vue`:

- Replace the `DC` text span with an image using the existing `company-brand__mark` class.
- Preserve the current 38 × 38 desktop footprint and 34 × 34 responsive footprint.
- Remove text-specific visual rules from the mark and use `display: block` with `object-fit: contain`. Do not add padding or crop the supplied transparent canvas.
- Keep the `DC-Agent` name, absolute positioning, spacing, and responsive behavior unchanged.

## Administration Application

In `admin-frontend/src/components/layout/AdminLayout.vue`:

- Replace the `DC` text span with the same image using the existing `admin-brand__mark` class.
- Preserve the current 38 × 38 footprint and sidebar spacing.
- Remove the blue text-tile styling and use `display: block` with `object-fit: contain` so the supplied logo colors remain unmodified.
- Keep the `DC-Agent` text, navigation link, route behavior, mobile layout, and accessible link label unchanged.

## Favicons

Add the following declaration to both `frontend/index.html` and `admin-frontend/index.html`:

```html
<link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />
```

The user application's existing local, uncommitted title change from `小D · 机密知识助手` to `DC智识中枢` must be preserved. The logo work must not overwrite, stage, or claim ownership of that title change.

## Testing

Automated tests protect:

- The user brand renders `/favicon-logo.svg`, keeps `DC-Agent`, and uses an image rather than a text node for `company-brand__mark`.
- The administration brand renders the same asset and keeps its existing navigation/accessibility contract.
- Both HTML entry files declare the SVG favicon.
- Both application builds include the public SVG without import or path errors.

Browser smoke verification checks the visible user and administration logos load with non-zero dimensions and the document favicon resolves successfully.

## Scope

In scope:

- The supplied SVG copied into both Vite public directories.
- Visible brand-mark replacement in the user and administration applications.
- Favicons for both applications.
- Focused regression tests and production builds.

Out of scope:

- Changing `DC-Agent` product text, page titles, colors elsewhere, navigation, or layout structure.
- Redrawing, recoloring, cropping, or otherwise editing the supplied SVG.
- Replacing unrelated icons such as navigation or status icons.
- Adding a shared package or changing Vite root configuration.

## Success Criteria

The supplied logo appears in both visible brand locations and both browser tabs, retains its original SVG colors and aspect ratio, loads in production builds, and does not disturb existing brand text, navigation, accessibility, responsive layout, or the user's uncommitted page-title change.
