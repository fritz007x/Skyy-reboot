// Finn in 3D: finn.glb (the mascot model with a skeleton added, see make_finn_model.py),
// drawn with three.js. He has two states:
//
//   idle      mouth closed, Finn still. The scene doesn't change, so nothing is redrawn.
//   speaking  while Skyy's voice plays:
//             - the jaw opens and closes with the sound actually playing (Voice.mouth,
//               voice.js), directly, so it stays in sync and stops on an interruption;
//             - the snout lifts a little as the jaw drops, as a real mouth opens both ways;
//             - the head nods on stressed syllables (sudden rises in loudness), with a
//               small random turn and tilt, and drifts slowly the rest of the time;
//             - the body leans in slightly and sways; the eye blinks every few seconds.
//             These ease in when Skyy starts and settle back to the rest pose when Skyy
//             stops, through damped springs, so the motion has a little weight.
//
// Finn.pause(true) holds him idle at once, whatever is playing. skyy.js calls it while the
// visitor speaks and Whisper transcribes, because Whisper runs on the same integrated GPU.
//
// For checking poses: ?open=0.5 holds the mouth half open, ?blink=1 shuts the eye,
// ?pitch=5 ?yaw=-5 ?roll=5 hold the head (degrees; pitch > 0 lifts the snout), and
// ?fps=60 changes the 30 fps limit while speaking (30 costs Whisper far less when the
// visitor interrupts Skyy).
import * as THREE from "three";
import { GLTFLoader } from "./vendor/jsm/loaders/GLTFLoader.js";

const params = new URLSearchParams(location.search);
const held = name => params.has(name) ? Number(params.get(name)) : null;
const fixedOpen = held("open");
const minFrameMs = 1000 / Number(params.get("fps") || 30);

const CAMERA_FOV = 28;                 // degrees
const MODEL_CENTRE = new THREE.Vector3(0, 50, 0);
const VIEW_HEIGHT = 112, VIEW_WIDTH = 125;    // model units to keep in view (he's 100 tall)

const deg = THREE.MathUtils.degToRad;
const JAW_MAX = deg(16);               // jaw angle with the mouth fully open
const LIFT = 0.3;                      // the snout rises by this share of the jaw's drop
// Slow drift while talking: two sine waves per axis, at unrelated rates so it never repeats
const DRIFT = {
  pitch: [[deg(1.2), 1.1, 0], [deg(0.6), 2.3, 1]],      // [amplitude, rad/s, phase]
  yaw: [[deg(3.0), 0.47, 2], [deg(1.2), 1.3, 0]],
  roll: [[deg(2.0), 0.61, 4], [deg(0.8), 1.7, 3]],
};
// Nods: a stressed syllable kicks the head's springs (speeds in rad/s)
const BEAT_RISE = 0.22;                // loudness jump over its recent average that counts
const BEAT_GAP = 0.3;                  // seconds, at least, between nods
const NOD_KICK = 1.0, TURN_KICK = 0.8, TILT_KICK = 0.8;
const HEAD_SPRING = [2.2, 0.45];       // [frequency Hz, damping]: a little overshoot
const EASE_SECS = 0.35;                // how fast the talking motion fades in and out
const LEAN = deg(1.2), SWAY = deg(0.6);
const STILL = 1e-4;                    // changes smaller than this (radians, 0..1) aren't drawn

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(1);             // the integrated GPU is shared with Whisper
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9bb0, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-40, 120, 100);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 10, 2000);

// Filled in when the model has loaded
let model = null;
let parts = null;    // {root, head, jaw, axes, tongue, eyelid}

// Animation state
const spring = () => ({ x: 0, v: 0 });
const head = { pitch: spring(), yaw: spring(), roll: spring() };
let talk = 0;                          // 0 idle .. 1 talking, eased
let average = 0, lastBeat = 0;
let state = "idle";
let paused = false;
let drawn = null;                      // the pose last drawn; null = draw again
let last = performance.now();

function fit() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  // Far enough to see all of him, whichever way the window is shaped
  const halfFov = deg(CAMERA_FOV) / 2;
  const distance = Math.max(VIEW_HEIGHT, VIEW_WIDTH / camera.aspect) / 2 / Math.tan(halfFov);
  camera.position.set(0, MODEL_CENTRE.y, distance);
  camera.lookAt(MODEL_CENTRE);
  camera.updateProjectionMatrix();
  drawn = null;
}
addEventListener("resize", fit);
fit();

new GLTFLoader().load("finn.glb", gltf => {
  model = gltf.scene;
  scene.add(model);
  model.traverse(part => { part.frustumCulled = false; });   // skinned: bounds move
  const bone = name => model.getObjectByName(name);
  const vector = v => new THREE.Vector3(...v);
  const headBone = bone("head"), jawBone = bone("jaw");
  parts = {
    head: headBone, jaw: jawBone,
    axes: {
      pitch: vector(headBone.userData.pitch), yaw: vector(headBone.userData.yaw),
      roll: vector(headBone.userData.roll), jaw: vector(jawBone.userData.axis),
    },
    tongue: model.getObjectByName("tongue"),
    eyelid: model.getObjectByName("eyelid"),
  };
  drawn = null;
}, undefined, err => {
  console.error("Couldn't load finn.glb:", err);
});

// Damped spring towards target, in small steps so it stays stable at any frame rate
function pull(s, target, dt) {
  const [hz, damping] = HEAD_SPRING;
  const w = 2 * Math.PI * hz;
  for (let left = dt; left > 0; left -= 1 / 120) {
    const h = Math.min(left, 1 / 120);
    s.v += (w * w * (target - s.x) - 2 * damping * w * s.v) * h;
    s.x += s.v * h;
  }
}

const drift = (waves, t) => waves.reduce((sum, [a, rate, phase]) => sum + a * Math.sin(t * rate + phase), 0);

function settle() {
  for (const s of Object.values(head)) s.x = s.v = 0;
  talk = 0;
}

const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();

function pose(p) {
  const { head: h, jaw, axes } = parts;
  h.quaternion.setFromAxisAngle(axes.yaw, p.yaw)
    .multiply(qa.setFromAxisAngle(axes.pitch, -p.pitch))   // + about SIDE tips the snout down
    .multiply(qb.setFromAxisAngle(axes.roll, p.roll));
  jaw.quaternion.setFromAxisAngle(axes.jaw, p.open * JAW_MAX);
  parts.tongue.morphTargetInfluences[0] = p.open;
  parts.eyelid.morphTargetInfluences[0] = p.blink;
  model.rotation.set(p.lean, 0, p.sway);                   // around his base
}

function frame(t) {
  requestAnimationFrame(frame);
  if (t - last < minFrameMs - 2) return;
  const dt = Math.min(0.1, (t - last) / 1000);
  last = t;
  const seconds = t / 1000;

  let open = 0, loud = 0, voice = 0;
  if (paused) {
    Voice.mouth.reset();
    settle();
  } else {
    ({ open, loud, voice } = Voice.mouth.step(dt, fixedOpen));
  }
  if (open < 0.002) open = 0;
  const speaking = !paused && (open > 0 || voice > 0.04);
  state = speaking ? "speaking" : "idle";
  talk += ((speaking ? 1 : 0) - talk) * (1 - Math.exp(-dt / EASE_SECS));

  // Nod on a sudden rise in loudness
  average += (loud - average) * (1 - Math.exp(-dt / 0.25));
  if (speaking && loud - average > BEAT_RISE && seconds - lastBeat > BEAT_GAP) {
    lastBeat = seconds;
    head.pitch.v -= NOD_KICK * (0.6 + 0.4 * Math.random());
    head.yaw.v += TURN_KICK * (Math.random() - 0.5);
    head.roll.v += TILT_KICK * (Math.random() - 0.5);
  }
  if (!paused) {
    pull(head.pitch, talk * drift(DRIFT.pitch, seconds) + LIFT * open * JAW_MAX, dt);
    pull(head.yaw, talk * drift(DRIFT.yaw, seconds), dt);
    pull(head.roll, talk * drift(DRIFT.roll, seconds), dt);
  }
  // Back at rest: snap the last tiny bit so the pose is exactly still and stops redrawing
  if (!speaking && talk < 0.02 && Object.values(head).every(s => Math.abs(s.x) < 1e-3 && Math.abs(s.v) < 1e-2)) {
    settle();
  }

  const p = {
    open,
    pitch: held("pitch") !== null ? deg(held("pitch")) : head.pitch.x,
    yaw: held("yaw") !== null ? deg(held("yaw")) : head.yaw.x,
    roll: held("roll") !== null ? deg(held("roll")) : head.roll.x,
    blink: held("blink") ?? (talk > 0.3 ? Voice.eye.blink(t) : 0),
    lean: talk * (LEAN + 0.4 * LEAN * Math.sin(seconds * 0.8)),
    sway: talk * SWAY * Math.sin(seconds * 0.53 + 1),
  };
  if (!parts) return;
  if (drawn && Object.keys(p).every(k => Math.abs(p[k] - drawn[k]) < STILL)) return;
  pose(p);
  renderer.render(scene, camera);
  drawn = p;
}
requestAnimationFrame(frame);

window.Finn = {
  attach: Voice.attach,
  pause(on) { paused = on; },
  get state() { return state; },
  get open() { return drawn ? drawn.open : 0; },
};
