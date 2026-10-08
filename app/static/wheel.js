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

  var canvas = null;
  var ctx = null;
  var entries = [];
  var rotation = 0;     // radians; 0 = segment 0's leading edge at 12 o'clock
  var spinToken = 0;

  function normalize(angle) {
    return ((angle % TAU) + TAU) % TAU;
  }

  function labelFor(label) {
    return label.length > LABEL_MAX ? label.slice(0, LABEL_MAX - 1) + "…" : label;
  }

  var thumbs = {};  // image_url -> Image once ready; false once failed

  /* Cached, fully-decoded thumbnail for url (null until then).
   * First sight starts the load; load/error redraw so the wheel settles
   * into photo or label-only without ever blocking the render. */
  function thumbFor(url) {
    if (!url) return null;
    if (url in thumbs) {
      var cached = thumbs[url];
      return cached && cached.complete && cached.naturalWidth > 0
        ? cached
        : null;
    }
    var img = new Image();
    thumbs[url] = img;
    img.onload = function () { draw(); };
    img.onerror = function () { thumbs[url] = false; draw(); };
    img.src = url;
    return null;
  }

  /* Draw one photo into a square at the segment's mid-angle, clipped to
   * that segment's wedge so neighbors stay untouched (spec §9). */
  function drawThumb(img, i, cx, cy, radius, delta) {
    var mid = (i + 0.5) * delta + rotation - HALF_PI;  // canvas angle
    var dist = radius * 0.62;
    // Square side: the wedge's chord at `dist`, never more than half r.
    var side = Math.min(radius * 0.5, 2 * dist * Math.sin(delta / 2) * 0.85);
    if (side < 12) return;  // crowded wheel: labels only, still readable
    var x = cx + Math.cos(mid) * dist;
    var y = cy + Math.sin(mid) * dist;
    // Contain-fit: the whole crop stays inside the square.
    var scale = Math.min(side / img.naturalWidth, side / img.naturalHeight);
    var w = img.naturalWidth * scale;
    var h = img.naturalHeight * scale;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, i * delta + rotation - HALF_PI,
            (i + 1) * delta + rotation - HALF_PI);
    ctx.closePath();
    ctx.clip();
    ctx.drawImage(img, x - w / 2, y - h / 2, w, h);
    ctx.restore();
  }

  function draw() {
    // Every OOB #wheel-pane swap replaces the node: re-acquire it per draw,
    // or we'd render into a detached canvas after the first swap.
    canvas = document.getElementById("wheel");
    ctx = canvas && canvas.getContext("2d");
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

    // Photos: each entry's image clipped into its own equal-width
    // segment. Until an image loads — or if it fails to load — the
    // segment simply keeps its palette fill and label.
    for (i = 0; i < n; i++) {
      var thumb = thumbFor(entries[i] && entries[i].image_url);
      if (thumb) drawThumb(thumb, i, cx, cy, radius, delta);
    }

    ctx.fillStyle = "#1f2430";
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    for (i = 0; i < n; i++) {
      var mid = (i + 0.5) * delta + rotation;  // angle from 12 o'clock
      var flipped = Math.sin(mid) < 0;         // keep left-side labels upright
      var text = labelFor(String(entries[i].label || ""));
      var px = flipped ? -radius * 0.55 : radius * 0.55;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(mid - HALF_PI);
      if (flipped) ctx.rotate(Math.PI);
      // Halo keeps the label readable when a photo fills the segment.
      ctx.strokeText(text, px, 0);
      ctx.fillText(text, px, 0);
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
