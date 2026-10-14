# Vendored chart JS: plotly.js (zero egress, PRD §3.4 / §5.2 / A13)

- Upstream: **plotly.js 4.1.1** (MIT, see `LICENSE`), the `plotly.min.js` bundled in the
  `plotly` 7.1.0 wheel on PyPI (`plotly/package_data/plotly.min.js`) — the same Python
  package the `[sandbox]` extra pins, so sandbox-made charts and this file match.
- Files: `plotly.min.js`, `LICENSE`.
- Integrity: both files' sha256 are pinned in `vdreamer.vendored.VENDORED_ASSETS`;
  `vdreamer doctor --offline` fails on a missing or modified file.
- Served at `/static/vendor/plotly/plotly.min.js` by the gateway. HTML artifacts load it
  with `<script src="/static/vendor/plotly/plotly.min.js">` (from Python:
  `fig.to_html(include_plotlyjs="/static/vendor/plotly/plotly.min.js")`).
- The bundle names tile servers for map traces; HTML artifacts are served under a CSP
  with `connect-src 'none'` and `img-src 'self' data:`, so a browser never reaches them.

To upgrade: copy `plotly.min.js` and the license from a newer `plotly` wheel, update the
versions above, and refresh the hashes.
