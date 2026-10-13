/* vdreamer docs shell: Swagger UI over the gateway's own OpenAPI 3.1 document.
 * validatorUrl: null keeps the badge validator from contacting its default
 * public host; the page CSP blocks every external origin regardless (PRD 5.2). */
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
