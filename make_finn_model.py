"""Make web/finn.glb, the Finn the kiosk page shows, from the mascot model.

    python make_finn_model.py "C:\\path\\to\\shark mascot 3d model.glb"

The source model (generated with Tripo) is a single closed skin with no bones, morph
targets or separate jaw: the grin, teeth and eye are painted on it. To let Finn talk, this:

1. Cuts the skin along the lower lip. LIP_LINE gives the lip's height around the head, read
   from a cylindrical unwrap of the model. Triangles crossing the line are split on it and
   the new vertices are duplicated, so the upper and lower lip can part cleanly.
2. Adds a skeleton of three bones, "body", "head" and "jaw" (a child of the head), which
   finn3d.js turns:
   - the jaw rotates around a hinge through the two mouth corners. It is what can be reached
     from the chin without crossing the cut, fading out towards the neck and past the corners;
   - the head turns around the neck. Its weights are 1 on the head, 0 on the body, shoulders
     and arms, and blend smoothly through the neck in between (solved as a smooth,
     "harmonic", function over the mesh, the usual way to make skinning weights).
   The rotation axes are stored in each bone's extras.
3. Adds the inside of the mouth: the same skin again with its triangles flipped, in dark
   maroon, so whatever shows through the parted lips is the inside of the head; a tongue
   (morph target "grow") that grows from nothing as the jaw drops; and an eyelid over the
   painted eye (morph target "blink") that sweeps down from a line along the top of the eye.
   In the rest pose with both morphs at 0 none of these can be seen, so the model looks
   exactly as it did.
4. Shrinks the 4096x4096 textures (about 260 MB of shared GPU memory with their mipmaps).
   Whisper runs on the same integrated GPU, and the page never shows that much detail.

Needs numpy, scipy and Pillow (all in the project's venv). The source file isn't modified.
"""
import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve

OUTPUT = Path(__file__).parent / "web" / "finn.glb"

# The head is turned from the body. Its frame: the axis the snout points along, and sideways.
HEAD_TURN = np.radians(32)
FORWARD = np.array([-np.sin(HEAD_TURN), 0, np.cos(HEAD_TURN)])
SIDE = np.array([np.cos(HEAD_TURN), 0, np.sin(HEAD_TURN)])
HEAD_CENTRE = np.array([2.6, 0, -0.4])
UNWRAP_AXIS = 5.0               # the unwrap's vertical axis, this far along FORWARD

# Where to cut, just above the top of the lower lip, so that the lip stays whole on the jaw:
# (degrees around the head, 0 = straight ahead; height in model units)
LIP_LINE = np.array([
    (-77, 56.5), (-75, 56.5), (-60, 55.4), (-50, 54.6), (-40, 53.0), (-30, 51.3), (-20, 50.3),
    (-10, 49.9), (0, 50.3), (10, 50.9), (21, 52.1), (27.5, 52.5), (40, 54.1), (48.75, 55.0),
    (60, 55.7), (77, 56.7),
])
CORNER = 77                     # degrees: the mouth corners, where the cut ends

NECK_TOP, NECK_BOTTOM = 46.0, 41.0   # the rotation fades out between these heights...
FADE_START, FADE_END = 70.0, 92.0    # ...and between these angles, past the mouth corners

MOUTH_COLOR = (0.035, 0.004, 0.008)       # linear RGB: the inside of the head
TONGUE_COLOR = (0.16, 0.018, 0.028)       # ...and the tongue, darker at its edges
# The tongue's middle, lying on the floor of the mouth (along FORWARD, height), half its
# width, length and thickness, and the point inside the jaw it grows from
TONGUE_CENTRE = (21.0, 50.0)
TONGUE_SIZE = (7.0, 4.5, 1.0)
TONGUE_START = (23.0, 47.5)

# The head turns around this point in the neck (along FORWARD, along SIDE, height). It
# follows the head fully from HEAD_FROM up and behind HEAD_BACK; the body stays still
# below BODY_BELOW, and below SHOULDER_TOP farther than SHOULDER_RADIUS from the neck.
NECK_PIVOT = (0.0, -2.0, 44.0)
HEAD_FROM, HEAD_BACK = 52.0, -28.0
BODY_BELOW = 36.0
SHOULDER_TOP, SHOULDER_RADIUS = 46.5, 21.5

# The painted eye on the side facing the camera: its middle (along FORWARD, height) and half
# its width and height, measured on the texture. The eyelid covers this oval.
EYE_CENTRE = (19.8, 78.2)
EYE_RADII = (4.9, 5.2)
LID_LIFT = 0.4                  # how far the eyelid sits off the skin
LID_ROWS, LID_COLUMNS = 8, 12

TEXTURE_SIZES = {"basecolor": 2048, "normal": 2048, "rm": 1024}
JPEG_QUALITY = 90


def read_glb(path):
    data = Path(path).read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    return gltf, data[20 + json_len + 8:]


def accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(binary, dtype, acc["count"] * width, start).reshape(-1, width)


def around_head(p):
    """Angle around the head (degrees) and distance from the unwrap axis."""
    rel = p - HEAD_CENTRE
    forward = rel @ FORWARD - UNWRAP_AXIS
    side = rel @ SIDE
    return np.degrees(np.arctan2(side, forward)), np.hypot(side, forward)


def above_lip(p):
    """Height above the lip line: > 0 upper head, < 0 jaw."""
    return p[..., 1] - np.interp(around_head(p)[0], LIP_LINE[:, 0], LIP_LINE[:, 1])


def cut_along_lip(pos, nor, uv, tris):
    """Split the triangles crossing the lip line, with separate vertices for each side.

    Returns the new arrays, one bool per vertex (True = below the cut) and the pairs of
    edges along the cut as (upper a, upper b, lower a, lower b).
    """
    angle = around_head(pos)[0]
    side = above_lip(pos)
    side = np.where(side == 0, 1e-5, side)    # no vertex exactly on the line
    pos, nor, uv = [list(a) for a in (pos, nor, uv)]
    below = list(side < 0)
    cache = {}

    def point_on(a, b, lower):
        """The vertex where edge a-b crosses the line, on the upper or the lower side."""
        key = (min(a, b), max(a, b), lower)
        if key not in cache:
            lo, hi = 0.0, 1.0                 # the line isn't straight in 3D: bisect
            for _ in range(24):
                mid = (lo + hi) / 2
                value = above_lip(pos[a] + mid * (pos[b] - pos[a]))
                if (value < 0) == bool(below[a]):
                    lo = mid
                else:
                    hi = mid
            t = (lo + hi) / 2
            normal = nor[a] + t * (nor[b] - nor[a])
            pos.append(pos[a] + t * (pos[b] - pos[a]))
            nor.append(normal / np.linalg.norm(normal))
            uv.append(uv[a] + t * (uv[b] - uv[a]))
            below.append(lower)
            cache[key] = len(pos) - 1
        return cache[key]

    out, cut_edges = [], []
    for tri in tris:
        flags = [bool(below[v]) for v in tri]
        spread = angle[tri].max() - angle[tri].min()
        if flags[0] == flags[1] == flags[2] or abs(angle[tri]).max() > CORNER or spread > 60:
            out.append(tri)
            continue
        # Rotate so that vertex 0 is the one alone on its side of the line
        k = next(i for i in range(3) if flags[i] != flags[(i + 1) % 3] and flags[i] != flags[(i + 2) % 3])
        v0, v1, v2 = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        alone = flags[k]
        p01, p02 = point_on(v0, v1, alone), point_on(v0, v2, alone)
        q01, q02 = point_on(v0, v1, not alone), point_on(v0, v2, not alone)
        out.append([v0, p01, p02])
        out.append([q01, v1, v2])
        out.append([q01, v2, q02])
        cut_edges.append((q01, q02, p01, p02) if alone else (p01, p02, q01, q02))
    return (np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32),
            np.array(out, np.uint32), np.array(below), np.array(cut_edges, np.uint32))


def jaw_weights(pos, tris, below):
    """1 on the jaw, fading out towards the neck and past the mouth corners, 0 elsewhere."""
    # The UV islands split the skin: join vertices at the same place, except across the cut
    keys = np.round(pos, 3)
    _, welded = np.unique(np.column_stack((keys, below)), axis=0, return_inverse=True)
    welded = welded.ravel()
    angle, radius = around_head(pos)
    allowed = below & (pos[:, 1] > NECK_BOTTOM) & (abs(angle) < FADE_END)

    neighbours = [[] for _ in range(welded.max() + 1)]
    for tri in tris:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if allowed[tri[a]] and allowed[tri[b]]:
                neighbours[welded[tri[a]]].append(welded[tri[b]])
                neighbours[welded[tri[b]]].append(welded[tri[a]])

    # Start from the middle of the chin and flood outwards
    chin = np.where(allowed & (abs(angle) < 10) & (pos[:, 1] > NECK_TOP))[0]
    seed = welded[chin[np.argmax(radius[chin])]]
    reached = {seed}
    stack = [seed]
    while stack:
        for n in neighbours[stack.pop()]:
            if n not in reached:
                reached.add(n)
                stack.append(n)
    on_jaw = allowed & np.isin(welded, np.fromiter(reached, int))

    smooth = lambda x: x * x * (3 - 2 * x)
    neck = smooth(np.clip((pos[:, 1] - NECK_BOTTOM) / (NECK_TOP - NECK_BOTTOM), 0, 1))
    past_corner = smooth(np.clip((FADE_END - abs(angle)) / (FADE_END - FADE_START), 0, 1))
    return on_jaw * neck * past_corner


def hinge_through_corners(pos):
    """The point and direction of the line through the two mouth corners."""
    ends = []
    for sign in (-1, 1):
        angle, radius = around_head(pos)
        near = abs(angle - sign * CORNER) < 3
        height = np.interp(sign * CORNER, LIP_LINE[:, 0], LIP_LINE[:, 1])
        near &= abs(pos[:, 1] - height) < 2
        ends.append(pos[near][np.argmax(radius[near])])
    a, b = ends
    return a, (b - a) / np.linalg.norm(b - a)


def tongue():
    """A flattened sphere that is a single point with "grow" at 0 and full size at 1."""
    rows, columns = 8, 16
    around = np.linspace(0, 2 * np.pi, columns, endpoint=False)
    # Unit sphere: rows from the back to the front of the tongue, sides around that axis
    points = [(0, 0, -1)]
    for along in np.linspace(-1, 1, rows + 2)[1:-1]:
        ring = np.sqrt(1 - along * along)
        points += [(ring * np.cos(a), ring * np.sin(a), along) for a in around]
    points.append((0, 0, 1))
    sphere = np.array(points)

    index = [(0, 1 + (i + 1) % columns, 1 + i) for i in range(columns)]
    for row in range(rows - 1):
        first = 1 + row * columns
        for i in range(columns):
            a, b = first + i, first + (i + 1) % columns
            index += [(a, b, a + columns), (b, b + columns, a + columns)]
    last = len(sphere) - 1
    first = 1 + (rows - 1) * columns
    index += [(first + i, first + (i + 1) % columns, last) for i in range(columns)]

    width, length, thickness = TONGUE_SIZE
    centre = HEAD_CENTRE + FORWARD * TONGUE_CENTRE[0] + np.array([0, TONGUE_CENTRE[1], 0])
    target = (centre + SIDE * (width * sphere[:, 0])[:, None]
              + np.array([0, 1, 0]) * (thickness * sphere[:, 1])[:, None]
              + FORWARD * (length * sphere[:, 2])[:, None])
    start = HEAD_CENTRE + FORWARD * TONGUE_START[0] + np.array([0, TONGUE_START[1], 0])
    closed = np.tile(start, (len(sphere), 1))
    shade = (0.45 + 0.55 * np.clip(sphere[:, 1] + 0.3, 0, 1))[:, None]
    return (closed.astype(np.float32), (target - closed).astype(np.float32),
            (shade * np.array(TONGUE_COLOR)).astype(np.float32), np.array(index, np.uint32).ravel())


def head_weights(pos, tris):
    """How much each vertex follows the head (rather than the body), 0..1."""
    _, welded = np.unique(np.round(pos, 3), axis=0, return_inverse=True)
    welded = welded.ravel()
    count = welded.max() + 1
    points = np.zeros((count, 3))
    points[welded] = pos
    edges = welded[tris][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    links = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    links = ((links + links.T) > 0).astype(float).tocsr()
    laplacian = (sparse.diags(np.asarray(links.sum(axis=1)).ravel()) - links).tocsr()

    rel = points - HEAD_CENTRE
    forward, side, height = rel @ FORWARD, rel @ SIDE, points[:, 1]
    from_neck = np.hypot(forward - NECK_PIVOT[0], side - NECK_PIVOT[1])
    head = (height >= HEAD_FROM) | ((forward < HEAD_BACK) & (height > NECK_PIVOT[2]))
    body = ~head & ((height <= BODY_BELOW) | ((height < SHOULDER_TOP) & (from_neck > SHOULDER_RADIUS)))
    fixed = head | body
    free = ~fixed
    weight = head.astype(float)
    # The smoothest blend between the fixed parts: the free weights' Laplacian is 0
    weight[free] = spsolve(laplacian[free][:, free].tocsc(),
                           -laplacian[free][:, fixed] @ weight[fixed])
    return np.clip(weight, 0, 1)[welded]


def raycast(origins, direction, pos, tris):
    """Where rays from each origin along direction first hit the triangles (Moller-Trumbore)."""
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = abs(det) > 1e-9
    a, e1, e2, p, det = a[ok], e1[ok], e2[ok], p[ok], det[ok]
    hits = []
    for origin in origins:
        s = origin - a
        u = (s * p).sum(1) / det
        q = np.cross(s, e1)
        v = (q @ direction) / det
        t = (q * e2).sum(1) / det
        good = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if not good.any():
            raise ValueError("a ray missed the head around the eye: check EYE_CENTRE")
        hits.append(origin + direction * t[good].min())
    return np.array(hits)


def eyelid(pos, nor, tris, uv, texture):
    """An oval lid over the painted eye, and the colour of the skin around it.

    Its rows are spread top to bottom over the oval. At rest every vertex sits on the top
    edge of the oval, above its own column, so the lid has no area until it blinks."""
    guess = HEAD_CENTRE + FORWARD * EYE_CENTRE[0] + np.array([0, EYE_CENTRE[1], 0])
    rel = pos - guess
    near = np.linalg.norm(rel - SIDE * (rel @ SIDE)[:, None], axis=1) < 9
    near &= (pos - HEAD_CENTRE) @ SIDE > 0
    patch = tris[near[tris].all(axis=1)]
    centre = raycast([guess + SIDE * 40], -SIDE, pos, patch)[0]
    close = near & (np.linalg.norm(pos - centre, axis=1) < 6)
    normal = nor[close].mean(axis=0)
    normal /= np.linalg.norm(normal)
    across = FORWARD - normal * (FORWARD @ normal)
    across /= np.linalg.norm(across)
    up = np.cross(normal, across)
    if up[1] < 0:
        up = -up

    rf, ry = EYE_RADII
    columns = np.linspace(-1, 1, LID_COLUMNS + 2)[1:-1] * rf
    half = ry * np.sqrt(1 - (columns / rf) ** 2)
    rows = np.linspace(0, 1, LID_ROWS)
    shut_2d = [(x, h - 2 * h * r) for r in rows for x, h in zip(columns, half)]
    rest_2d = [(x, h) for r in rows for x, h in zip(columns, half)]

    def on_skin(points):
        origins = [centre + across * x + up * y + normal * 10 for x, y in points]
        return raycast(origins, -normal, pos, patch) + normal * LID_LIFT

    shut, rest = on_skin(shut_2d), on_skin(rest_2d)
    index = []
    for r in range(LID_ROWS - 1):
        for k in range(LID_COLUMNS - 1):
            a = r * LID_COLUMNS + k
            index += [a, a + LID_COLUMNS, a + 1, a + 1, a + LID_COLUMNS, a + LID_COLUMNS + 1]
    tri = np.array(index).reshape(-1, 3)
    if np.cross(shut[tri[0, 1]] - shut[tri[0, 0]], shut[tri[0, 2]] - shut[tri[0, 0]]) @ normal < 0:
        tri = tri[:, ::-1]

    # The lid takes the colour of the skin in a ring just outside the eye
    samples = []
    rng = np.random.default_rng(0)
    for t in patch:
        for _ in range(6):
            w = rng.dirichlet((1, 1, 1))
            point = w @ pos[t] - centre
            if np.hypot(point @ across / rf, point @ up / ry) < 1.25 or np.hypot(point @ across / rf, point @ up / ry) > 1.6:
                continue
            u, v = w @ uv[t]
            samples.append(texture[int(v * (texture.shape[0] - 1)), int(u * (texture.shape[1] - 1))])
    srgb = np.median(np.array(samples), axis=0) / 255
    color = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    normals = np.tile(normal, (len(rest), 1))
    return (rest.astype(np.float32), (shut - rest).astype(np.float32), normals.astype(np.float32),
            tri.astype(np.uint32).ravel(), color, centre)


class GlbWriter:
    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors = [], []

    def add_view(self, data, target=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.views.append(view)
        self.blob += data
        return len(self.views) - 1

    def add(self, array, kind, target=34962):
        component = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125,
                     np.dtype(np.uint8): 5121}[array.dtype]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[kind]
        entry = {"bufferView": self.add_view(array.tobytes(), target),
                 "componentType": component, "count": array.size // width, "type": kind}
        if kind == "VEC3":
            entry["min"] = array.min(axis=0).tolist()
            entry["max"] = array.max(axis=0).tolist()
        self.accessors.append(entry)
        return len(self.accessors) - 1

    def save(self, path, gltf):
        gltf["bufferViews"] = self.views
        gltf["accessors"] = self.accessors
        gltf["buffers"] = [{"byteLength": len(self.blob)}]
        text = json.dumps(gltf, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        blob = bytes(self.blob) + b"\0" * (-len(self.blob) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(blob)))
            f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
            f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def unlit(name, color):
    return {"name": name, "extensions": {"KHR_materials_unlit": {}},
            "pbrMetallicRoughness": {"baseColorFactor": color, "metallicFactor": 0, "roughnessFactor": 1}}


def base_color(gltf, binary):
    image = next(i for i in gltf["images"] if "basecolor" in i["name"])
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB"))


def shrink_texture(gltf, binary, image):
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    picture = Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB")
    size = next(s for name, s in TEXTURE_SIZES.items() if name in image["name"])
    picture = picture.resize((size, size), Image.LANCZOS)
    out = io.BytesIO()
    picture.save(out, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the mascot .glb")
    args = parser.parse_args()

    gltf, binary = read_glb(args.source)
    primitive = gltf["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    pos0 = accessor(gltf, binary, attributes["POSITION"])
    nor0 = accessor(gltf, binary, attributes["NORMAL"])
    uv0 = accessor(gltf, binary, attributes["TEXCOORD_0"])
    tris0 = accessor(gltf, binary, primitive["indices"]).reshape(-1, 3)

    pos, nor, uv, tris, below, cut_edges = cut_along_lip(pos0, nor0, uv0, tris0)
    jaw = jaw_weights(pos, tris, below)
    head = head_weights(pos, tris)
    hinge, axis = hinge_through_corners(pos)
    # Point the hinge axis so that a positive angle lowers the front of the jaw
    front = HEAD_CENTRE + FORWARD * 30 + np.array([0, 50, 0])
    if np.cross(axis, front - hinge)[1] > 0:
        axis = -axis
    neck = HEAD_CENTRE + FORWARD * NECK_PIVOT[0] + SIDE * NECK_PIVOT[1] + np.array([0, NECK_PIVOT[2], 0])

    # Bones 0 body, 1 head, 2 jaw; the jaw's share of a vertex is taken from the head's
    joints = np.tile(np.array([0, 1, 2, 0], np.uint8), (len(pos), 1))
    weights = np.column_stack((1 - head, head * (1 - jaw), head * jaw, np.zeros(len(pos)))).astype(np.float32)

    texture = base_color(gltf, binary)
    tongue_pos, tongue_deltas, tongue_colors, tongue_index = tongue()
    lid_pos, lid_deltas, lid_normals, lid_index, lid_color, eye = eyelid(pos, nor, tris, uv, texture)

    writer = GlbWriter()

    def on_bone(bone, count):
        return (writer.add(np.tile(np.array([bone, 0, 0, 0], np.uint8), (count, 1)), "VEC4"),
                writer.add(np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)), "VEC4"))

    skin = {
        "attributes": {"POSITION": writer.add(pos, "VEC3"), "NORMAL": writer.add(nor, "VEC3"),
                       "TEXCOORD_0": writer.add(uv, "VEC2"),
                       "JOINTS_0": writer.add(joints, "VEC4"), "WEIGHTS_0": writer.add(weights, "VEC4")},
        "indices": writer.add(tris.ravel(), "SCALAR", 34963),
        "material": 0, "mode": 4,
    }
    inside = {
        "attributes": {key: skin["attributes"][key] for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")},
        "indices": writer.add(tris[:, ::-1].ravel(), "SCALAR", 34963),
        "material": 1, "mode": 4,
    }
    joints_t, weights_t = on_bone(2, len(tongue_pos))
    tongue_part = {
        "attributes": {"POSITION": writer.add(tongue_pos, "VEC3"),
                       "COLOR_0": writer.add(tongue_colors, "VEC3"),
                       "JOINTS_0": joints_t, "WEIGHTS_0": weights_t},
        "indices": writer.add(tongue_index, "SCALAR", 34963),
        "material": 2, "mode": 4,
        "targets": [{"POSITION": writer.add(tongue_deltas, "VEC3")}],
    }
    joints_l, weights_l = on_bone(1, len(lid_pos))
    lid_part = {
        "attributes": {"POSITION": writer.add(lid_pos, "VEC3"), "NORMAL": writer.add(lid_normals, "VEC3"),
                       "JOINTS_0": joints_l, "WEIGHTS_0": weights_l},
        "indices": writer.add(lid_index, "SCALAR", 34963),
        "material": 3, "mode": 4,
        "targets": [{"POSITION": writer.add(lid_deltas, "VEC3")}],
    }
    # The bones sit at their pivots with no rotation, so their inverse bind matrices only
    # move the pivot back to the origin (column-major 4x4)
    pivots = [np.zeros(3), neck, hinge]
    inverse_bind = np.array([np.r_[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -p, 1] for p in pivots], np.float32)

    images = []
    for image in gltf["images"]:
        images.append({"name": image["name"].rsplit(".", 1)[0], "mimeType": "image/jpeg",
                       "bufferView": writer.add_view(shrink_texture(gltf, binary, image))})

    out = {
        "asset": {"version": "2.0", "generator": "make_finn_model.py, from a Tripo model"},
        "extensionsUsed": ["KHR_materials_specular", "KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Finn", "children": [1, 4, 5, 6]},
            {"name": "body", "children": [2]},
            {"name": "head", "translation": neck.tolist(), "children": [3],
             "extras": {"pitch": SIDE.tolist(), "yaw": [0, 1, 0], "roll": FORWARD.tolist()}},
            {"name": "jaw", "translation": (hinge - neck).tolist(), "extras": {"axis": axis.tolist()}},
            {"name": "skin", "mesh": 0, "skin": 0},
            {"name": "tongue", "mesh": 1, "skin": 0},
            {"name": "eyelid", "mesh": 2, "skin": 0},
        ],
        "skins": [{"joints": [1, 2, 3], "skeleton": 1,
                   "inverseBindMatrices": writer.add(inverse_bind.ravel(), "MAT4", None)}],
        "meshes": [{"name": "skin", "primitives": [skin, inside]},
                   {"name": "tongue", "primitives": [tongue_part], "weights": [0],
                    "extras": {"targetNames": ["grow"]}},
                   {"name": "eyelid", "primitives": [lid_part], "weights": [0],
                    "extras": {"targetNames": ["blink"]}}],
        "materials": [gltf["materials"][0] | {"name": "skin"},
                      unlit("inside of the mouth", [*MOUTH_COLOR, 1]),
                      unlit("tongue", [1, 1, 1, 1]),
                      {"name": "eyelid", "pbrMetallicRoughness": {
                          "baseColorFactor": [*np.round(lid_color, 4).tolist(), 1],
                          "metallicFactor": 0, "roughnessFactor": 0.35}}],
        "textures": gltf["textures"], "samplers": gltf["samplers"], "images": images,
    }
    writer.save(OUTPUT, out)

    print(f"{len(pos0)} vertices -> {len(pos)}; {len(cut_edges)} edges along the cut; "
          f"{(jaw > 0.5).sum()} on the jaw, {(head > 0.5).sum()} on the head")
    print(f"jaw hinge at {hinge.round(1).tolist()}, axis {axis.round(2).tolist()}; "
          f"neck at {neck.round(1).tolist()}; eye at {eye.round(1).tolist()}, "
          f"lid colour {np.round(lid_color, 3).tolist()}")
    print(f"{OUTPUT}: {OUTPUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
