/* vdreamer docs shell: Swagger UI over the gateway's own OpenAPI 3.1 document.
 * validatorUrl: null keeps the badge validator from contacting its default
 * public host; the page CSP blocks every external origin regardless (zero egress). */
/* Follow the viewer's light/dark preference with Swagger UI's own .dark-mode theme
 * (shipped in the vendored CSS); without it a dark browser shows Swagger's light-theme
 * text on a dark canvas. */
(function () {
  var dark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function apply() {
    document.documentElement.classList.toggle("dark-mode", !!(dark && dark.matches));
  }
  apply();
  if (dark && dark.addEventListener) {
    dark.addEventListener("change", apply);
  }
})();

window.addEventListener("load", function () {
  window.ui = SwaggerUIBundle({
    url: "/api/v1/openapi.json",
    dom_id: "#swagger-ui",
    validatorUrl: null,
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis],
    layout: "BaseLayout"
  });
});
