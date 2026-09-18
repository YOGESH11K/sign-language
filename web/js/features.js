/**
 * Feature extraction mirror of app/feature_extractor.py.
 *
 * Same layout and math so the browser runs the identical 142-dim features the
 * Python model was trained on:
 *   per hand: 63 (21 landmarks rel to wrist / hand size) + 5 fingertip dists
 *             + 1 spread + 2 handedness one-hot = 71
 *   frame     = Left hand block + Right hand block    = 142
 */
export const PER_HAND_DIM = 71;
export const FEATURE_DIM = PER_HAND_DIM * 2;
export const NUM_LANDMARKS = 21;

const TIP_IDS = [4, 8, 12, 16, 20];
const WRIST = 0;
const MIDDLE_MCP = 9;
const INDEX_TIP = 8;
const RING_TIP = 16;
const HANDEDNESS_ORDER = { Left: 0, Right: 1, Unknown: 2 };

function norm3(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** landmarks: array of 21 [x,y,z]. handedness: "Left"|"Right"|"Unknown". */
export function handFeature(landmarks, handedness) {
  const f = new Float32Array(PER_HAND_DIM);
  if (!landmarks || landmarks.length < NUM_LANDMARKS) return f;
  const lm = landmarks;
  const wrist = lm[WRIST];
  const scale = norm3(lm[MIDDLE_MCP], wrist);
  if (scale < 1e-6) return f;

  let i = 0;
  for (let k = 0; k < NUM_LANDMARKS; k++) {
    f[i++] = (lm[k][0] - wrist[0]) / scale;
    f[i++] = (lm[k][1] - wrist[1]) / scale;
    f[i++] = (lm[k][2] - wrist[2]) / scale;
  }
  for (let t = 0; t < TIP_IDS.length; t++) {       // 63..67 fingertip dists
    f[i++] = norm3(lm[TIP_IDS[t]], wrist) / scale;
  }
  f[i++] = norm3(lm[INDEX_TIP], lm[RING_TIP]) / scale;  // 68 spread
  if (handedness === "Right") f[i++] = 1.0;         // 69
  else if (handedness === "Left") f[i++] = 1.0;     // 70
  return f;
}

/** hands: [{landmarks:[[x,y,z]*21], handedness}]. Returns Float32Array(142). */
export function frameFeatures(hands) {
  const ordered = hands
    .map(h => ({ h, k: HANDEDNESS_ORDER[h.handedness] ?? 2 }))
    .sort((a, b) => a.k - b.k)
    .map(o => o.h);
  const vecs = ordered.slice(0, 2).map(h => handFeature(h.landmarks, h.handedness));
  while (vecs.length < 2) vecs.push(new Float32Array(PER_HAND_DIM));
  const out = new Float32Array(FEATURE_DIM);
  vecs.forEach((v, j) => out.set(v, j * PER_HAND_DIM));
  return out;
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const ex = logits.map(v => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / s);
}