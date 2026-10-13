# Vendored Swagger UI (zero egress, PRD §5.2 / A13)

- Upstream: `swagger-ui-dist` **5.33.1** (Apache-2.0, see `LICENSE`), taken from the
  `drf-spectacular-sidecar` 2026.10.1 wheel on PyPI (a package that exists only to
  ship `swagger-ui-dist` for offline use).
- Files: `swagger-ui-bundle.js`, `swagger-ui.css`, `favicon-32x32.png`, `LICENSE`.
  Source maps are not vendored; the CSS `sourceMappingURL` comment was stripped.
- Integrity: every file's sha256 is pinned in `vdreamer.vendored.VENDORED_ASSETS`;
  `vdreamer doctor --offline` fails on a missing or modified file.
- Served at `/static/vendor/swagger-ui/` by the gateway; `/api/v1/docs` loads them under a
  CSP with no external origins, and `swagger-init.js` sets `validatorUrl: null` so the
  badge validator (`validator.swagger.io`) is never contacted.

To upgrade: copy the same four files from a newer `swagger-ui-dist`, strip the CSS
source-map comment, update the version above, and refresh the hashes.
