/* Wheelio wheel: draw equal-width segments and animate to the server's index.
 *
 * The server picks the random winner — this file only renders entries and
 * spins the canvas to the data-pre-index the server reports in #wheel-data.
 */
(function () {
  "use strict";

  var TAU = Math.PI * 2;
  var HALF_PI = Math.PI / 2;
  var PALETTE = [
    "#ffd9d9", "#ffe6cc", "#fff3cc", "#ddf5d5",
    "#d2f0e4", "#d3ecfb", "#ddd8ff", "#f4d9ff"
  ];
  var LABEL_MAX = 18;   // draw-time truncation only; the data stays intact
  var TURNS = 5;        // extra full turns per spin
  var DURATION = 3500;  // ms: ≈5 full turns + settle

  var canvas = document.getElementById("wheel");
  var ctx = canvas && canvas.getContext("2d");
  var entries = [];
  var rotation = 0;     // radians; 0 = segment 0's leading edge at 12 o'clock
  var spinToken = 0;

  function normalize(angle) {
    return ((angle % TAU) + TAU) % TAU;
  }

  function labelFor(label) {
    return label.length > LABEL_MAX ? label.slice(0, LABEL_MAX - 1) + "…" : label;
  }

  function draw() {
    if (!ctx) return;
    var w = canvas.width;
    var h = canvas.height;
    var cx = w / 2;
    var cy = h / 2;
    var radius = Math.min(w, h) / 2 - 2;
    var n = entries.length;
    var i;

    ctx.clearRect(0, 0, w, h);
    if (!n) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius - 1, 0, TAU);
      ctx.strokeStyle = "#d9dce3";
      ctx.setLineDash([6, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    var delta = TAU / n;
    for (i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, i * delta + rotation - HALF_PI,
              (i + 1) * delta + rotation - HALF_PI);
      ctx.closePath();
      ctx.fillStyle = PALETTE[i % PALETTE.length];
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }

    ctx.fillStyle = "#1f2430";
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (i = 0; i < n; i++) {
      var mid = (i + 0.5) * delta + rotation;  // angle from 12 o'clock
      var flipped = Math.sin(mid) < 0;         // keep left-side labels upright
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(mid - HALF_PI);
      if (flipped) ctx.rotate(Math.PI);
      ctx.fillText(labelFor(String(entries[i].label || "")),
                   flipped ? -radius * 0.55 : radius * 0.55, 0);
      ctx.restore();
    }
  }

  window.WheelIO = {
    /* Draw the given {label, image_url} entries at the current rotation. */
    render: function (next) {
      entries = Array.isArray(next) ? next : [];
      draw();
    },

    /* Animate ≈5 turns + settle so `index`'s center lands at 12 o'clock. */
    spinTo: function (index, done) {
      var n = entries.length;
      if (!n) {
        if (done) done();
        return;
      }
      var token = ++spinToken;
      var delta = TAU / n;
      var desired = normalize(-(index + 0.5) * delta);
      var start = rotation;
      var target = start + TURNS * TAU + normalize(desired - start);
      var beganAt = null;

      requestAnimationFrame(function frame(now) {
        if (token !== spinToken) return;  // a newer spin superseded this one
        if (beganAt === null) beganAt = now;
        var t = Math.min((now - beganAt) / DURATION, 1);
        rotation = start + (target - start) * (1 - Math.pow(1 - t, 3));
        draw();
        if (t < 1) {
          requestAnimationFrame(frame);
          return;
        }
        rotation = normalize(target);
        draw();
        if (done) done();
      });
    }
  };

  function parseEntries(raw) {
    try {
      var parsed = JSON.parse(raw || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }

  /* After any htmx request, re-read #wheel-data (inside the swapped
     #wheel-pane): spin when the server says it picked a winner, otherwise
     just redraw. Never computes a winner itself. */
  document.addEventListener("htmx:afterRequest", function () {
    var wheelData = document.getElementById("wheel-data");
    if (!wheelData) return;
    var data = wheelData.dataset;
    var next = parseEntries(data.entries);
    if (data.animate === "1") {
      wheelData.removeAttribute("data-animate");  // consume; one spin per swap
      var index = parseInt(data.preIndex, 10);
      WheelIO.spinTo(isNaN(index) ? 0 : index, function () {
        WheelIO.render(next);
      });
    } else {
      WheelIO.render(next);
    }
  });

  var initial = document.getElementById("wheel-data");
  if (initial) WheelIO.render(parseEntries(initial.dataset.entries));
})();
