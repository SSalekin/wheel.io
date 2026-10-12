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
  var MAX_THUMBS = 150; // photo cache cap: oldest entries evict first

  var canvas = null;
  var ctx = null;
  var entries = [];
  var rotation = 0;     // radians; 0 = segment 0's leading edge at 12 o'clock
  var spinToken = 0;
  var winningIndex = null;
  var wheelSize = 480;  // logical (CSS) px; re-read when the canvas is swapped
  var lastCanvas = null;
  var confetti = [];
  var confettiRunning = false;  // at most one animation loop ever runs

  function normalize(angle) {
    return ((angle % TAU) + TAU) % TAU;
  }

  function labelFor(label) {
    // Array.from splits by code point: truncation never halves a
    // surrogate pair (emoji names stay whole).
    var chars = Array.from(label);
    return chars.length > LABEL_MAX
      ? chars.slice(0, LABEL_MAX - 1).join("") + "…"
      : label;
  }

  var thumbs = new Map();  // image_url -> Image once ready; false once failed

  /* Cached, fully-decoded thumbnail for url (null until then).
   * First sight starts the load; load/error redraw so the wheel settles
   * into photo or label-only without ever blocking the render. The cache
   * is bounded (MAX_THUMBS): the oldest URL evicts first. */
  function thumbFor(url) {
    if (!url) return null;
    if (thumbs.has(url)) {
      var cached = thumbs.get(url);
      return cached && cached.complete && cached.naturalWidth > 0
        ? cached
        : null;
    }
    var img = new Image();
    thumbs.set(url, img);
    if (thumbs.size > MAX_THUMBS) thumbs.delete(thumbs.keys().next().value);
    img.onload = function () { draw(); };
    img.onerror = function () { thumbs.set(url, false); draw(); };
    img.src = url;
    return null;
  }

  /* True when the new entry list still shows at least one photo the old
   * one showed — i.e. the cache is still wanted. A wheel replacement
   * (new photos, or photos → text) shares nothing and drops the cache. */
  function sharesPhotoWith(before, after) {
    for (var i = 0; i < after.length; i++) {
      var url = after[i] && after[i].image_url;
      if (url) {
        for (var j = 0; j < before.length; j++) {
          if (before[j] && before[j].image_url === url) return true;
        }
      }
    }
    return false;
  }

  /* Draw one photo into a square at the segment's mid-angle, clipped to
   * that segment's wedge so neighbors stay untouched (spec §9).
   * Returns true when a photo was drawn; false when the segment is too
   * crowded for one (the caller falls back to the label). */
  function drawThumb(img, i, cx, cy, radius, delta) {
    var mid = (i + 0.5) * delta + rotation - HALF_PI;  // canvas angle
    var dist = radius * 0.62;
    // Square side: the wedge's chord at `dist`, never more than half r.
    var side = Math.min(radius * 0.5, 2 * dist * Math.sin(delta / 2) * 0.85);
    if (side < 12) return false;  // crowded wheel: label instead, still readable
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
    return true;
  }

  function draw() {
    // Every OOB #wheel-pane swap replaces the node: re-acquire it per draw,
    // or we'd render into a detached canvas after the first swap.
    canvas = document.getElementById("wheel");
    if (!canvas) return;
    if (canvas !== lastCanvas) {
      // A fresh node ships the logical size as its attributes.
      lastCanvas = canvas;
      wheelSize = parseInt(canvas.getAttribute("width"), 10) || 480;
    }
    // HiDPI: the backing store follows the display density; all drawing
    // below stays in CSS pixels via the dpr transform.
    var dpr = window.devicePixelRatio || 1;
    if (canvas.width !== wheelSize * dpr) {
      canvas.width = Math.round(wheelSize * dpr);
      canvas.height = Math.round(wheelSize * dpr);
    }
    ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var w = wheelSize;
    var h = wheelSize;
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
    // segment simply keeps its palette fill and label. A photo too
    // crowded to fit leaves photoDrawn[i] false so its label shows.
    var photoDrawn = [];
    for (i = 0; i < n; i++) {
      var thumb = thumbFor(entries[i] && entries[i].image_url);
      photoDrawn[i] = thumb ? drawThumb(thumb, i, cx, cy, radius, delta) : false;
    }

    // Winning segment glow
    if (winningIndex !== undefined && winningIndex !== null && winningIndex >= 0 && winningIndex < n) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, winningIndex * delta + rotation - HALF_PI,
              (winningIndex + 1) * delta + rotation - HALF_PI);
      ctx.closePath();
      ctx.fillStyle = "rgba(79, 70, 229, 0.25)";
      ctx.fill();
      ctx.strokeStyle = "#4f46e5";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(79, 70, 229, 0.6)";
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.restore();
    }

    // Center hub
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, TAU);
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = "rgba(0,0,0,0.15)";
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Pointer at top
    ctx.beginPath();
    ctx.moveTo(cx, 8);
    ctx.lineTo(cx - 10, 28);
    ctx.lineTo(cx + 10, 28);
    ctx.closePath();
    ctx.fillStyle = "#ef4444";
    ctx.shadowColor = "rgba(0,0,0,0.2)";
    ctx.shadowBlur = 4;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#1f2430";
    // Font scales with the room the segment gives (arc length at the
    // label radius): crowded wheels shrink labels instead of overlapping.
    var fontPx = Math.max(8, Math.min(13, delta * radius * 0.2));
    ctx.font = "600 " + fontPx + "px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    for (i = 0; i < n; i++) {
      // Skip the label when a photo was drawn — the face crop is enough.
      if (photoDrawn[i]) continue;
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

  /* The Go button node is replaced by OOB swaps mid-spin, so the busy
   * state and its undo always act on the CURRENT node — and the undo
   * only touches a button still saying "Spinning..." (the server may
   * have re-rendered it disabled for an emptied wheel). */
  function setGoBusy(busy) {
    var btn = document.querySelector(".go-button");
    if (!btn) return;
    if (busy) {
      btn.disabled = true;
      btn.textContent = "Spinning...";
    } else if (btn.textContent === "Spinning...") {
      btn.disabled = false;
      btn.textContent = "Go";
    }
  }

  window.WheelIO = {
    /* Draw the given {label, image_url} entries at the current rotation.
     * `glow` (an index or null) marks the winning segment in these
     * entries; omitting it clears any previous glow. */
    render: function (next, glow) {
      var before = entries;
      entries = Array.isArray(next) ? next : [];
      // A wheel replacement shares no photos with the old wheel: drop
      // the cache instead of hoarding dead Image objects.
      if (!sharesPhotoWith(before, entries)) thumbs.clear();
      winningIndex = glow === undefined ? null : glow;
      // Wait a frame so the swapped canvas is in the DOM before drawing.
      requestAnimationFrame(function () {
        draw();
      });
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

      setGoBusy(true);

      // Reduced motion: land on the same segment without the flight —
      // no animation, no confetti storm.
      if (window.matchMedia &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        rotation = normalize(target);
        draw();
        setGoBusy(false);
        if (done) done();
        return;
      }

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
        launchConfetti();
        setGoBusy(false);
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

  /* Simple confetti burst — repeated bursts extend the running loop
   * instead of stacking a second one. */
  function launchConfetti() {
    var colors = ["#4f46e5", "#ffd9d9", "#ffe6cc", "#ddf5d5", "#d3ecfb", "#f4d9ff"];
    for (var i = 0; i < 40; i++) {
      confetti.push({
        x: wheelSize / 2,
        y: wheelSize / 2,
        vx: (Math.random() - 0.5) * 8,
        vy: (Math.random() - 0.5) * 8 - 3,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: Math.random() * 6 + 3,
        life: 1
      });
    }
    if (!confettiRunning) {
      confettiRunning = true;
      requestAnimationFrame(drawConfetti);
    }
  }

  function drawConfetti() {
    if (!confetti.length) {
      confettiRunning = false;
      return;
    }
    draw();
    for (var i = confetti.length - 1; i >= 0; i--) {
      var c = confetti[i];
      c.x += c.vx;
      c.y += c.vy;
      c.vy += 0.15;
      c.life -= 0.015;
      if (c.life <= 0) {
        confetti.splice(i, 1);
        continue;
      }
      ctx.globalAlpha = c.life;
      ctx.fillStyle = c.color;
      ctx.fillRect(c.x, c.y, c.size, c.size);
    }
    ctx.globalAlpha = 1;
    if (confetti.length) requestAnimationFrame(drawConfetti);
  }

  /* Re-fetch the picked sidebar from the server. The spin responses no
     longer ship it OOB — the winner should appear only once the wheel
     has stopped, so the done-callback of the spin asks for it. */
  function refreshPicked() {
    var tokenInput = document.querySelector('#chat-form input[name="token"]');
    if (!tokenInput || !window.fetch) return;
    fetch("/picked?token=" + encodeURIComponent(tokenInput.value))
      .then(function (res) { return res.text(); })
      .then(function (html) {
        var fresh = document.createRange
          ? document.createRange().createContextualFragment(html).querySelector("#picked-list")
          : null;
        var current = document.getElementById("picked-list");
        if (fresh && current) current.replaceWith(fresh);
      })
      .catch(function () { /* stale token / network hiccup: keep the sidebar as-is */ });
  }

  /* After any htmx swap, re-read #wheel-data (inside the swapped
     #wheel-pane): spin when the server says it picked a winner, otherwise
     just redraw. Never computes a winner itself. */
  var lastDataNode = null;
  document.addEventListener("htmx:afterSwap", function (event) {
    var wheelData = document.getElementById("wheel-data");
    if (!wheelData || wheelData === lastDataNode) return;
    lastDataNode = wheelData;
    var data = wheelData.dataset;
    var next = parseEntries(data.entries);
    if (data.animate === "1") {
      wheelData.removeAttribute("data-animate");  // consume; one spin per swap
      var index = parseInt(data.preIndex, 10);
      // The server spun the entries it shipped as data-spin-entries; the
      // pointer must land on the winner even when remove_picked already
      // popped it from the shipped wheel.
      var spinEntries = data.spinEntries ? parseEntries(data.spinEntries) : next;
      var glow =
        data.glowIndex === undefined ? null : parseInt(data.glowIndex, 10);
      WheelIO.render(spinEntries);
      WheelIO.spinTo(isNaN(index) ? 0 : index, function () {
        // The final entries land only once the wheel stops, glow included.
        WheelIO.render(next, isNaN(glow) ? null : glow);
        refreshPicked();  // winner lands in the sidebar only when the wheel stops
      });
    } else {
      WheelIO.render(next);
    }
  });

  var initial = document.getElementById("wheel-data");
  if (initial) {
    lastDataNode = initial;
    WheelIO.render(parseEntries(initial.dataset.entries));
  }

  /* File preview thumbnails in the chat bar. Every preview is an object
   * URL; each one is revoked when its preview dies (remove, re-select,
   * submit) so the blobs don't outlive the page. */
  (function () {
    var fileInput = document.getElementById("file-input");
    var previewContainer = document.getElementById("file-preview");
    if (!fileInput || !previewContainer) return;

    var previewUrls = [];  // object URLs currently shown

    function clearPreviews() {
      // Revoking an already-revoked URL is a no-op, so remove-clicks
      // may revoke early without bookkeeping here.
      previewUrls.forEach(function (url) { URL.revokeObjectURL(url); });
      previewUrls = [];
      previewContainer.innerHTML = "";
    }

    fileInput.addEventListener("change", function () {
      clearPreviews();
      var files = Array.prototype.slice.call(fileInput.files);
      files.forEach(function (file, index) {
        if (!file.type.startsWith("image/")) return;
        var url = URL.createObjectURL(file);
        previewUrls.push(url);
        var item = document.createElement("div");
        item.className = "file-preview-item";
        var img = document.createElement("img");
        img.src = url;
        img.alt = file.name;
        var removeBtn = document.createElement("button");
        removeBtn.className = "file-preview-remove";
        removeBtn.innerHTML = "&times;";
        removeBtn.title = "Remove";
        removeBtn.addEventListener("click", function () {
          URL.revokeObjectURL(url);
          var dt = new DataTransfer();
          var remaining = Array.prototype.slice.call(fileInput.files).filter(function (_, i) { return i !== index; });
          remaining.forEach(function (f) { dt.items.add(f); });
          fileInput.files = dt.files;
          item.remove();
          if (!previewContainer.children.length) previewContainer.innerHTML = "";
        });
        item.appendChild(img);
        item.appendChild(removeBtn);
        previewContainer.appendChild(item);
      });
    });

    document.getElementById("chat-form").addEventListener("submit", function () {
      clearPreviews();
    });

    // Clear the picker only on success: a failed request keeps the
    // files so the user can retry them.
    document.getElementById("chat-form").addEventListener("htmx:afterRequest", function (e) {
      if (e.detail && e.detail.successful) fileInput.value = "";
    });
  })();

  /* Enter to send, Shift+Enter for newline */
  (function () {
    var chatInput = document.getElementById("chat-input");
    if (!chatInput) return;
    chatInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        document.getElementById("chat-form").requestSubmit();
      }
    });
  })();

  /* Auto-scroll chat to bottom on new message */
  (function () {
    var chatLog = document.getElementById("chat-log");
    if (!chatLog) return;
    var observer = new MutationObserver(function () {
      // Follow the conversation, but don't yank the view when the user
      // scrolled up to read older messages.
      var nearBottom =
        chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 80;
      if (nearBottom) chatLog.scrollTop = chatLog.scrollHeight;
    });
    observer.observe(chatLog, { childList: true });
  })();
})();
