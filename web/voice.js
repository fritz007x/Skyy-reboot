// What both Finns (finn.js = 2D, finn3d.js = 3D) take from Skyy's voice as it plays, and
// the eye behaviour of the 2D one.
//
// Voice.attach(input) analyses the audio that is playing: the WebRTC voice stream, or an
// <audio> element playing a file (the ?demo page, see skyy.js).
//
// Voice.mouth.step(dt) returns how open the mouth is (0..1) and a slower voice envelope:
//   loudness (RMS) opens the mouth; the spectrum shapes it. Hissing sounds ("s", "sh",
//   "f": energy above 3.5 kHz) keep the teeth almost together, front vowels ("ee", "eh":
//   more energy at 0.9-3 kHz) open less than "ah"/"oh" (energy below 0.9 kHz), and a
//   sudden drop in level snaps the jaw shut, as on "p", "b", "m". Because it follows the
//   sound actually being played, the mouth stops when Skyy is interrupted.
// Voice.eye.blink(t) is 0 (open) .. 1 (shut): a blink every few seconds, sometimes two.
// Voice.eye.gaze(t, talking, dt) is where the eye looks, x and y in -1..1: around the
// room when idle, ahead while Skyy talks.
const Voice = (() => {
  const GATE = 0.012;             // RMS below this is silence
  const FULL = 0.22;              // RMS at which the jaw is fully open (tuned on the Piper voices)
  // Smoothing rates: share of the gap to the target closed per 1/60 s (see follow())
  const OPEN_RATE = 0.6;          // mouth opening...
  const CLOSE_RATE = 0.3;         // ...when closing...
  const SNAP_RATE = 0.7;          // ...and when the level suddenly drops (p, b, m)

  const clamp01 = x => Math.min(1, Math.max(0, x));
  const ramp = (lo, hi, x) => clamp01((x - lo) / (hi - lo));
  // Move `value` towards `target` at `rate` per 1/60 s, the same on 60 and 120 Hz screens
  const follow = (value, target, rate, dt) => value + (target - value) * (1 - (1 - rate) ** (dt * 60));

  let ctx = null, source = null, analyser = null, samples = null, spectrum = null;

  // Called on every (re)connection with the new voice stream, or once with an <audio>
  // element in demo mode
  function attach(input) {
    if (!ctx) {
      ctx = new AudioContext();
      analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.3;
      samples = new Float32Array(analyser.fftSize);
      spectrum = new Float32Array(analyser.frequencyBinCount);
    }
    if (source) source.disconnect();
    if (input instanceof HTMLMediaElement) {
      // An element routed through Web Audio only sounds if it goes on to the speakers
      source = ctx.createMediaElementSource(input);
      source.connect(ctx.destination);
    } else {
      // Not connected to the speakers: the <audio> element plays the voice
      source = ctx.createMediaStreamSource(input);
    }
    source.connect(analyser);
    if (ctx.state === "suspended") ctx.resume();
  }

  // Loudness 0..1 and the share of energy in the mid and high bands
  function listen() {
    if (!analyser) return { loud: 0, mid: 0, high: 0 };
    analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const s of samples) sum += s * s;
    const loud = ramp(GATE, FULL, Math.sqrt(sum / samples.length));
    if (loud === 0) return { loud, mid: 0, high: 0 };

    analyser.getFloatFrequencyData(spectrum);
    const hz = ctx.sampleRate / analyser.fftSize;
    let low = 0, mid = 0, high = 0;
    for (let i = 1; i < spectrum.length; i++) {
      const f = i * hz, p = 10 ** (spectrum[i] / 10);
      if (f >= 150 && f < 900) low += p;
      else if (f >= 900 && f < 3000) mid += p;
      else if (f >= 3500 && f < 8000) high += p;
    }
    const total = low + mid + high || 1;
    return { loud, mid: mid / total, high: high / total };
  }

  const mouth = {
    open: 0,      // smoothed mouth opening, 0..1
    voice: 0,     // slower voice envelope, for the scene
    loud: 0,      // loudness right now, 0..1
    prevLoud: 0,
    // fixed: hold the mouth at this opening (the ?open= check), or null
    step(dt, fixed = null) {
      const { loud, mid, high } = listen();
      let target;
      if (fixed !== null) target = fixed;
      else {
        const hiss = ramp(0.25, 0.6, high);         // s, sh, f: teeth together
        const front = ramp(0.1, 0.4, mid);          // ee, eh: smaller opening
        target = loud * (1 - 0.85 * hiss) * (1 - 0.35 * front);
      }
      const rate = target > this.open ? OPEN_RATE
                 : (this.prevLoud - loud > 0.25 ? SNAP_RATE : CLOSE_RATE);
      this.open = follow(this.open, target, rate, dt);
      this.prevLoud = this.loud = loud;
      this.voice = follow(this.voice, loud, loud > this.voice ? 0.2 : 0.05, dt);
      return this;
    },
    reset() { this.open = 0; },
  };

  const eye = {
    next: performance.now() + 2000, start: -1, double: false,
    look: [0, 0], target: [0, 0], nextGlance: 0,

    blink(t) {
      // a blink closes in 70 ms, holds 40 ms, opens in 90 ms
      if (this.start < 0 && t >= this.next) this.start = t;
      if (this.start < 0) return 0;
      const e = t - this.start;
      if (e > 200) {
        this.start = -1;
        this.double = !this.double && Math.random() < 0.2;
        this.next = t + (this.double ? 120 : 2500 + Math.random() * 4000);
        return 0;
      }
      return e < 70 ? e / 70 : e < 110 ? 1 : 1 - (e - 110) / 90;
    },

    gaze(t, talking, dt) {
      if (talking) this.target = [-0.75, 0.3];     // looks ahead (Finn faces left)
      else if (t > this.nextGlance) {              // idle: look somewhere else now and then
        this.target = [Math.random() * 2 - 1, Math.random() * 2 - 1];
        this.nextGlance = t + 1500 + Math.random() * 3500;
      }
      this.look = this.look.map((g, i) => follow(g, this.target[i], 0.15, dt));
      return this.look;
    },
  };

  return { attach, mouth, eye, ramp };
})();
