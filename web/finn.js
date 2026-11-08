// The 2D Finn: the MDC_Finn.jpg picture, animated from Skyy's voice as it plays (voice.js
// turns the audio into mouth opening, blinks and glances; the 3D Finn uses the same).
//
//   Mouth  the lower jaw is cut out of the picture and rotates around the back corner of
//          the mouth, with the dark inside and the tongue showing as it opens.
//   Eye    blinks (an eyelid closes over it) and the highlight in the pupil moves.
//   Scene  the sunburst turns slowly and brightens with the voice, a glow behind Finn
//          pulses as he talks, and Finn breathes, sways a little and nods on syllables.
//
// Finn.pause(true) freezes the whole scene. skyy.js calls it while the visitor speaks and
// Whisper transcribes: Whisper runs on the same integrated GPU that draws the page, and a
// moving Finn made it ~20% slower (+30-70 ms per turn), a still one didn't.
//
// ?debug in the URL draws the jaw outline and hinge, ?open=9 holds the jaw open 9 degrees
// (for checking the cut-out), ?fps=30 limits how often the scene is redrawn (the GPU
// drawing it is the one Whisper runs on).
const Finn = (() => {
  const HINGE = [1006, 480];      // back corner of the mouth; the jaw rotates around it
  const FRONT = [676, 558];       // front end of the cut between the teeth
  const SUN = [960, 542];         // centre of the sunburst
  const RAYS = 36;                // light rays, one every 360/RAYS degrees
  const MAX_ANGLE = 9;            // degrees, jaw fully open
  const SPIN = 0.8;               // sunburst rotation, degrees per second

  const params = new URLSearchParams(location.search);
  const fixedAngle = params.has("open") ? Number(params.get("open")) : null;
  const minFrameMs = params.has("fps") ? 1000 / Number(params.get("fps")) : 0;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = id => document.getElementById(id);
  const svg = $("finn").ownerSVGElement;
  const finn = $("finn"), jaw = $("jaw"), mouth = $("mouth"), tongue = $("tongue");
  const rays = $("rays"), glow = $("glow");
  const glint = $("glint"), lid = $("lid"), lidLine = $("lid-line");

  // --- the sunburst rays: light wedges from the centre, 5 degrees wide every 10
  {
    const R = 2400, step = 360 / RAYS;
    let d = "";
    for (let i = 0; i < RAYS; i++) {
      const a0 = i * step * Math.PI / 180, a1 = (i * step + step / 2) * Math.PI / 180;
      d += `M${SUN}L${SUN[0] + R * Math.cos(a0)},${SUN[1] + R * Math.sin(a0)}` +
           `L${SUN[0] + R * Math.cos(a1)},${SUN[1] + R * Math.sin(a1)}Z`;
    }
    const path = document.createElementNS(svg.namespaceURI, "path");
    path.setAttribute("d", d);
    rays.appendChild(path);
  }

  const ramp = Voice.ramp;

  function rotate([x, y], deg) {
    const a = deg * Math.PI / 180, dx = x - HINGE[0], dy = y - HINGE[1];
    return [HINGE[0] + dx * Math.cos(a) - dy * Math.sin(a),
            HINGE[1] + dx * Math.sin(a) + dy * Math.cos(a)];
  }

  function setJaw(deg) {
    // Negative = counter-clockwise on screen, so the front of the jaw drops
    jaw.setAttribute("transform", `rotate(${-deg} ${HINGE})`);
    if (deg < 0.2) {
      mouth.setAttribute("d", "");
      tongue.setAttribute("opacity", 0);
      return;
    }
    const low = rotate(FRONT, -deg);
    // Front edge slightly curved inwards, like the inside of the lips
    const cx = (FRONT[0] + low[0]) / 2 + 12, cy = (FRONT[1] + low[1]) / 2;
    mouth.setAttribute("d", `M${FRONT} L${HINGE} L${low} Q${cx},${cy} ${FRONT}Z`);
    // The tongue rests on the lower jaw, just above its (rotated) top edge
    const [tx, ty] = rotate([800, 536], -deg);
    tongue.setAttribute("cx", tx);
    tongue.setAttribute("cy", ty - 4);
    tongue.setAttribute("opacity", ramp(3, 7, deg));
  }

  // --- eye: blinks and glances
  function setEye(t, talking, dt) {
    const shut = still ? 0 : Voice.eye.blink(t);
    lid.setAttribute("visibility", shut > 0 ? "visible" : "hidden");
    lid.setAttribute("transform", `translate(0 226) scale(1 ${Math.max(0.001, shut)}) translate(0 -226)`);
    lidLine.setAttribute("opacity", ramp(0.75, 1, shut));
    const [x, y] = Voice.eye.gaze(t, talking, dt);
    glint.setAttribute("cx", 881 + 4 * x);
    glint.setAttribute("cy", 262 + 3 * y);
  }

  // --- main loop
  let spin = 0;
  let last = performance.now();
  let paused = false;

  function pause(on) {
    if (on === paused) return;
    paused = on;
    if (on) {                       // freeze with the mouth closed and the eye open
      Voice.mouth.reset();
      setJaw(0);
      lid.setAttribute("visibility", "hidden");
      lidLine.setAttribute("opacity", 0);
    }
  }

  function frame(t) {
    if (paused || t - last < minFrameMs - 2) {
      if (paused) last = t;         // no jump in the animations when it resumes
      requestAnimationFrame(frame);
      return;
    }
    const dt = Math.min(0.1, (t - last) / 1000);
    last = t;

    const { open: level, voice } = Voice.mouth.step(dt, fixedAngle !== null ? fixedAngle / MAX_ANGLE : null);
    setJaw(level * MAX_ANGLE);
    setEye(t, voice > 0.05, dt);

    if (!still) {
      spin = (spin + SPIN * dt) % 360;
      rays.setAttribute("transform", `rotate(${spin} ${SUN})`);
      rays.setAttribute("opacity", 0.8 + 0.2 * voice);
      glow.setAttribute("opacity", 0.04 + 0.45 * voice);

      // Breathing and a slow sway from the bottom of the picture; a nod while talking
      const breath = Math.sin(t / 1600), sway = Math.sin(t / 2300) * 0.4;
      const scale = 1.008 + 0.006 * breath;
      const nod = level * 6;
      finn.setAttribute("transform",
        `translate(960 1080) rotate(${sway}) scale(${scale}) translate(-960 ${-1080 + nod})`);
    }
    requestAnimationFrame(frame);
  }

  if (params.has("debug")) {
    const outline = $("jaw-outline").cloneNode();
    outline.removeAttribute("id");
    outline.setAttribute("class", "debug");
    jaw.appendChild(outline);
    const dot = document.createElementNS(svg.namespaceURI, "circle");
    dot.setAttribute("cx", HINGE[0]); dot.setAttribute("cy", HINGE[1]); dot.setAttribute("r", 6);
    dot.setAttribute("fill", "#0f0");
    svg.appendChild(dot);
  }

  requestAnimationFrame(frame);
  return { attach: Voice.attach, pause };
})();
