"""Genora grey boardomatic: builds the whole scene in Blender from scratch.

    blender -b --factory-startup -P build_scene.py -- <out.blend>

Timeline follows the Magnific prompt (24 fps, 15 s = 360 frames):
  0-3 s   Genora puffs compressed air and lifts off
  3-8 s   flies over the row, the jets knock every slab down (they crack)
  8-12 s  drops onto the road, runs the half loop upside down, shoots off the open end
  12-15 s rolls upright over the broken, glitching slabs and lands in smoke

Two scenes share the same objects:
  Boardomatic_Cuts     multi-angle edit (markers switch cameras SH01..SH08)
  Boardomatic_OneTake  one continuous handheld "phone in the stands" camera
"""
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(ROOT, "textures")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else os.path.join(ROOT, "genora_boardomatic.blend")
# "grey": motion-reference build for Seedance. No logos, no brand colors, no red LEDs or
# pink smoke, so the video model takes only motion and camera from it and the look from the assets.
GREY = "grey" in argv[1:]

FPS = 24
FRAMES = 360
rnd = random.Random(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = "Boardomatic_Cuts"
SET = bpy.data.collections.new("GENORA_SET")
scene.collection.children.link(SET)


# ---------------------------------------------------------------- helpers

def srgb(*c):
    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return tuple(lin(v) for v in c)


def clamp(u, a=0.0, b=1.0):
    return max(a, min(b, u))


def smooth(u):
    u = clamp(u)
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def material(name, color, rough=0.5, metal=0.0, emit=0.0, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = emit
    if image:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(TEX, image), check_existing=True)
        tex.location = (-400, 200)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if emit:
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        nt.nodes.active = tex
    m.diffuse_color = (*color, 1)
    m.roughness = rough
    m.metallic = metal
    return m


def link(name, me, mat=None, parent=None):
    ob = bpy.data.objects.new(name, me)
    SET.objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), parent=None, size=1.0):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = size
    ob.location = loc
    SET.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def box_mesh(name, dims):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * dims[0], v.co.y * dims[1], v.co.z * dims[2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def rounded_box(name, dims, radius, mat, parent=None, segs=4):
    me = box_mesh(name, dims)
    for p in me.polygons:
        p.use_smooth = True
    ob = link(name, me, mat, parent)
    bev = ob.modifiers.new("Bevel", 'BEVEL')
    bev.width = radius
    bev.segments = segs
    bev.limit_method = 'NONE'
    bev.harden_normals = True
    return ob


def cone_mesh(name, r_bottom, r_top, depth, segs=20, z_shift=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r_bottom,
                          radius2=r_top, depth=depth)
    for v in bm.verts:
        v.co.z += z_shift
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def decal(name, w, h, mat, parent, loc, facing="-Y"):
    if facing == "-Y":
        verts = [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)]
    else:  # +Z
        verts = [(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for i, c in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = c
    ob = link(name, me, mat, parent)
    ob.location = loc
    return ob


def add_beam(bm, p1, p2, t):
    d = p2 - p1
    if d.length < 1e-4:
        return
    d.normalize()
    a = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
    u = d.cross(a).normalized() * (t / 2)
    v = d.cross(u).normalized() * (t / 2)
    corners = [u + v, -u + v, -u - v, u - v]
    vs = [bm.verts.new(p1 + c) for c in corners] + [bm.verts.new(p2 + c) for c in corners]
    for q in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0), (4, 5, 6, 7)):
        bm.faces.new([vs[i] for i in q])


def add_box(bm, center, size, rot=Matrix.Identity(3)):
    sx, sy, sz = (s / 2 for s in size)
    pts = [Vector((x, y, z)) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
    vs = [bm.verts.new(center + rot @ p) for p in pts]
    for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        bm.faces.new([vs[i] for i in q])


# --- animation helpers (dense per-frame baking) ---

def _fcurves(idb):
    act = idb.animation_data.action
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        cb = action_get_channelbag_for_slot(act, idb.animation_data.action_slot)
        if cb is not None:
            return cb.fcurves
    except ImportError:
        pass
    return act.fcurves


IP = {'CONSTANT': 0, 'LINEAR': 1, 'BEZIER': 2}


def bake(idb, path, frames, values, interp='LINEAR'):
    """values: list of tuples (one per frame) or scalars."""
    n_idx = len(values[0]) if isinstance(values[0], (tuple, list, Vector, Quaternion)) else 0
    idxs = range(n_idx) if n_idx else [0]
    curves = []
    for i in idxs:
        if n_idx:
            idb.keyframe_insert(path, index=i, frame=frames[0])
        else:
            idb.keyframe_insert(path, frame=frames[0])
        fc = _fcurves(idb).find(path, index=i)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        co = []
        for f, v in zip(frames, values):
            co.extend((f, v[i] if n_idx else v))
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [IP[interp]] * len(frames))
        fc.update()
        curves.append(fc)
    return curves


def quat_tn(T, N):
    X = T.normalized()
    Z = N.normalized()
    Y = Z.cross(X)
    return Matrix((X, Y, Z)).transposed().to_quaternion()


def fix_signs(quats):
    out = [quats[0]]
    for q in quats[1:]:
        if out[-1].dot(q) < 0:
            q = -q
        out.append(q)
    return out


ALL_FRAMES = list(range(1, FRAMES + 1))


# ---------------------------------------------------------------- materials

M = {
    "ground": material("Ground", srgb(74, 76, 80), rough=0.9),
    "road": material("Road", srgb(46, 47, 51), rough=0.8),
    "white_line": material("WhiteLine", srgb(225, 225, 225), emit=1.0),
    "led": material("RedLED", srgb(255, 30, 40), emit=12.0),
    "truss": material("Truss", srgb(150, 152, 156), rough=0.4, metal=0.8),
    "mast": material("Mast", srgb(120, 122, 126), rough=0.5, metal=0.6),
    "lamp": material("FloodLamp", srgb(240, 246, 255), emit=40.0),
    "genora_body": material("GenoraBody", srgb(22, 22, 24), rough=0.08),
    "genora_top": material("GenoraTop", (1, 1, 1), emit=4.0, image="genora_top.png"),
    "chrome": material("Chrome", srgb(200, 202, 206), rough=0.08, metal=1.0),
    "jet": material("Jet", srgb(170, 215, 255), emit=8.0),
    "vapor": material("Vapor", srgb(236, 234, 238), rough=1.0),
    "smoke": material("ImpactSmoke", srgb(240, 160, 165), rough=1.0),
}
for key, alpha in (("vapor", 0.55), ("smoke", 0.6), ("jet", 0.75)):
    M[key].diffuse_color[3] = alpha
    M[key].surface_render_method = 'BLENDED'



# ---------------------------------------------------------------- track + half loop

W = 7.0                      # road width
XSTART, X0 = -70.0, 36.0     # road start / where the ramp starts bending up
R1, V, R2 = 7.0, 5.0, 6.0    # bottom arc, vertical straight, top arc
XE = 25.0                    # open end of the top section (points left, over the row)
CX, CZ = X0 + R1 - R2, R1 + V
TOPZ = CZ + R2
S_A1 = X0 - XSTART
S_V = S_A1 + R1 * math.pi / 2
S_A2 = S_V + V
S_TOP = S_A2 + R2 * math.pi / 2
S_END = S_TOP + (CX - XE)
Y = Vector((0, 1, 0))


def road_at(s):
    """Centerline point, tangent and road normal at arc length s."""
    if s <= S_A1:
        return Vector((XSTART + s, 0, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))
    if s <= S_V:
        p = (s - S_A1) / R1
        return (Vector((X0 + R1 * math.sin(p), 0, R1 - R1 * math.cos(p))),
                Vector((math.cos(p), 0, math.sin(p))), Vector((-math.sin(p), 0, math.cos(p))))
    if s <= S_A2:
        return Vector((X0 + R1, 0, R1 + (s - S_V))), Vector((0, 0, 1)), Vector((-1, 0, 0))
    if s <= S_TOP:
        p = (s - S_A2) / R2
        return (Vector((CX + R2 * math.cos(p), 0, CZ + R2 * math.sin(p))),
                Vector((-math.sin(p), 0, math.cos(p))), Vector((-math.cos(p), 0, -math.sin(p))))
    return Vector((CX - (s - S_TOP), 0, TOPZ)), Vector((-1, 0, 0)), Vector((0, 0, -1))


def stations(s0, s1):
    out = []
    s = s0
    while s < s1:
        out.append(s)
        s += 2.0 if s < S_A1 - 1 else 0.25
    out.append(s1)
    for b in (S_A1, S_V, S_A2, S_TOP):
        if s0 < b < s1:
            out.append(b)
    return sorted(set(round(x, 5) for x in out))


def sweep(name, s0, s1, y0, y1, n0, n1, mat):
    verts, faces = [], []
    ss = stations(s0, s1)
    for s in ss:
        P, T, N = road_at(s)
        for (yy, nn) in ((y0, n0), (y1, n0), (y1, n1), (y0, n1)):
            verts.append(P + Y * yy + N * nn)
    for i in range(len(ss) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, a + k2, b + k2, b + k))
    faces.append((3, 2, 1, 0))
    e = 4 * (len(ss) - 1)
    faces.append((e, e + 1, e + 2, e + 3))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    return link(name, me, mat)


track = empty("TRACK", size=0.1)
ground = link("Ground", box_mesh("Ground", (900, 900, 0.02)), M["ground"])
ground.location = (0, 0, -0.011)
ground.parent = track

for nm, args in {
    "Road": (-W / 2, W / 2, -0.4, 0.02, M["road"]),
    "CenterLine": (-0.08, 0.08, 0.02, 0.035, M["white_line"]),
    "LED_Left": (W / 2 - 0.36, W / 2 - 0.16, 0.02, 0.08, M["led"]),
    "LED_Right": (-W / 2 + 0.16, -W / 2 + 0.36, 0.02, 0.08, M["led"]),
}.items():
    sweep(nm, 0.0, S_END, *args).parent = track

# extra red LED lines painted on the asphalt next to the road (as in the style frames)
for side, nm in ((1, "LED_Ground_Far"), (-1, "LED_Ground_Near")):
    yy = side * (W / 2 + 3.0)
    sweep(nm, 0.0, S_A1 + 6, min(yy, yy + 0.2), max(yy, yy + 0.2), 0.0, 0.05, M["led"]).parent = track

# steel truss along the outside of the ramp (open end, no back tower)
bm = bmesh.new()
pts = []
s = S_A1 + 1.2
while s < S_END - 0.05:
    pts.append(s)
    s += 1.6
pts.append(S_END - 0.05)
prev = None
for s in pts:
    P, T, N = road_at(s)
    st = {
        "iL": P + Y * (W / 2) - N * 0.4, "iR": P - Y * (W / 2) - N * 0.4,
        "oL": P + Y * (W / 2) - N * 2.3, "oR": P - Y * (W / 2) - N * 2.3,
    }
    add_beam(bm, st["iL"], st["oL"], 0.12)
    add_beam(bm, st["iR"], st["oR"], 0.12)
    add_beam(bm, st["oL"], st["oR"], 0.12)
    if prev:
        for k in st:
            add_beam(bm, prev[k], st[k], 0.18)
        add_beam(bm, prev["iL"], st["oL"], 0.08)
        add_beam(bm, prev["iR"], st["oR"], 0.08)
        add_beam(bm, prev["oL"], st["oR"], 0.08)
    prev = st
me = bpy.data.meshes.new("Truss")
bm.to_mesh(me)
bm.free()
link("Truss", me, M["truss"]).parent = track

# floodlight masts behind the track
for i, (mx, my) in enumerate([(-14, 30), (6, 36), (28, 40), (50, 32), (-34, 20), (70, 18)]):
    root = empty(f"Floodlight_{i}", (mx, my, 0), parent=track)
    pole = link(f"FloodPole_{i}", cone_mesh("Pole", 0.24, 0.24, 22, 10, 11), M["mast"], root)
    d = (Vector((16, 0, 0)) - Vector((mx, my, 0)))
    d.z = 0
    d.normalize()
    rot = (Matrix.Rotation(math.atan2(d.x, -d.y), 3, 'Z') @ Matrix.Rotation(math.radians(14), 3, 'X'))
    hb = bmesh.new()
    add_box(hb, Vector((0, 0, 22.5)), (4.6, 0.5, 3.4), rot)
    head = bpy.data.meshes.new("FloodHead")
    hb.to_mesh(head)
    hb.free()
    link(f"FloodHead_{i}", head, M["mast"], root)
    lb = bmesh.new()
    for gx in range(6):
        for gz in range(5):
            c = Vector(((gx - 2.5) * 0.7, -0.3, (gz - 2) * 0.62))
            add_box(lb, Vector((0, 0, 22.5)) + rot @ c, (0.5, 0.12, 0.44), rot)
    lamps = bpy.data.meshes.new("FloodLamps")
    lb.to_mesh(lamps)
    lb.free()
    link(f"FloodLamps_{i}", lamps, M["lamp"], root)


# ---------------------------------------------------------------- Genora

genora = empty("GENORA", size=2.0)
genora.rotation_mode = 'QUATERNION'
G_BODY = (3.0, 3.0, 0.75)
rounded_box("Genora_Body", G_BODY, 0.42, M["genora_body"], genora, segs=5)
decal("Genora_TopLogo", 2.45, 2.45, M["genora_top"], genora, (0, 0, G_BODY[2] / 2 + 0.006), facing="+Z")
NOZ = [(1.08, 1.08), (1.08, -1.08), (-1.08, 1.08), (-1.08, -1.08)]
noz_me = cone_mesh("Nozzle", 0.36, 0.27, 0.5, 20, -0.25)
jet_me = cone_mesh("Jet", 0.36, 0.22, 1.0, 20, -0.5)
jets = []
for i, (nx, ny) in enumerate(NOZ):
    n = link(f"Genora_Nozzle_{i}", noz_me, None, genora)
    n.location = (nx, ny, -G_BODY[2] / 2 + 0.05)
    if not noz_me.materials:
        noz_me.materials.append(M["chrome"])
    j = link(f"Genora_Jet_{i}", jet_me, None, genora)
    j.location = (nx, ny, -G_BODY[2] / 2 - 0.45)
    if not jet_me.materials:
        jet_me.materials.append(M["jet"])
    jets.append(j)
HOVER = 0.85   # body center above the road surface when resting on its nozzles

# --- flight path, frame by frame
Q_UP_FWD = Quaternion()                                            # upright, heading +X
Q_UP_BACK = quat_tn(Vector((-1, 0, 0)), Vector((0, 0, 1)))         # upright, heading -X
Q_UD_BACK = quat_tn(Vector((-1, 0, 0)), Vector((0, 0, -1)))        # upside down, heading -X
PUFFS = (26, 38, 50)
LIFT0, FLY0, FLY1, DROP1, RAMP1 = 56, 72, 192, 212, 270
FLY_Z = 6.4


def fly_x(f):
    u = clamp((f - FLY0) / (FLY1 - FLY0))
    return 1.0 + 32.0 * (0.6 * u + 0.4 * smooth(u))


# speed profile on the ramp: accelerate up the loop, bleed speed over the top
ramp_s0 = 35.5 - XSTART
prof = []
for f in range(DROP1, RAMP1 + 1):
    u = (f - DROP1) / (RAMP1 - DROP1)
    prof.append(0.15 + (math.sin(math.pi / 2 * min(u / 0.55, 1)) if u < 0.55 else lerp(1.0, 0.55, (u - 0.55) / 0.45)))
cum = [0.0]
for v in prof[1:]:
    cum.append(cum[-1] + v)
ramp_s = {f: ramp_s0 + (S_END - ramp_s0) * cum[i] / cum[-1] for i, f in enumerate(range(DROP1, RAMP1 + 1))}
exit_speed = (ramp_s[RAMP1] - ramp_s[RAMP1 - 1])

gpos, grot = [], []
EXIT = Vector((XE, 0, TOPZ - HOVER))
for f in ALL_FRAMES:
    pitch = 0.0
    q = Q_UP_FWD
    if f <= LIFT0:
        bump = sum(0.14 * math.sin(math.pi * (f - p) / 10) for p in PUFFS if p <= f < p + 10)
        pos = Vector((0, 0, HOVER + bump))
    elif f <= FLY0:
        u = (f - LIFT0) / (FLY0 - LIFT0)
        pos = Vector((1.0 * u * u, 0, lerp(HOVER, FLY_Z, smooth(u))))
        pitch = math.radians(6) * u
    elif f <= FLY1:
        u = (f - FLY0) / (FLY1 - FLY0)
        pos = Vector((fly_x(f), 0, FLY_Z + 0.25 * math.sin(u * 4 * math.pi)))
        pitch = math.radians(6)
    elif f <= DROP1:
        u = (f - FLY1) / (DROP1 - FLY1)
        pos = Vector((lerp(33.0, 35.5, u), 0, lerp(FLY_Z, HOVER, smooth(u))))
        pitch = math.radians(6) * (1 - smooth(u))
    elif f <= RAMP1:
        P, T, N = road_at(ramp_s[f])
        pos = P + N * HOVER
        q = quat_tn(T, N)
    elif f <= 292:                      # shoots off the open end, upside down
        k = f - RAMP1
        x = XE - (exit_speed * k - 0.5 * (exit_speed - 0.32) / 22 * k * k)
        pos = Vector((x, 0, EXIT.z + 0.9 * math.sin(math.pi * k / 30) - 0.004 * k * k))
        q = Q_UD_BACK
    elif f <= 312:                      # 180 degree roll to come upright
        u = (f - 292) / 20
        p0 = gpos[291 - 1]
        pos = Vector((p0.x - 0.3 * (f - 292), 0, lerp(p0.z, 12.0, smooth(u))))
        q = Q_UD_BACK @ Quaternion((1, 0, 0), math.pi * smooth(u))
    elif f <= 344:                      # descends over the broken row and lands
        u = (f - 312) / 32
        p0 = gpos[311]
        pos = Vector((lerp(p0.x, 2.5, smooth(u)), 0, lerp(p0.z, HOVER, 1 - (1 - u) ** 2.2)))
        q = Q_UP_BACK
        pitch = -math.radians(10) * math.sin(math.pi * u)
    else:
        k = f - 344
        pos = Vector((2.5, 0, HOVER + 0.16 * math.exp(-k / 4) * abs(math.sin(k * 0.55))))
        q = Q_UP_BACK
    gpos.append(pos)
    grot.append(q @ Quaternion((0, 1, 0), pitch))
grot = fix_signs(grot)
bake(genora, "location", ALL_FRAMES, [tuple(p) for p in gpos])
bake(genora, "rotation_quaternion", ALL_FRAMES, [tuple(q) for q in grot])


def jet_len(f):
    for p in PUFFS:
        if p <= f < p + 5:
            return 1.7 * math.sin(math.pi * (f - p) / 5) + 0.05
    if LIFT0 <= f < FLY0:
        return 2.2
    if FLY0 <= f < FLY1:
        return 1.7
    if FLY1 <= f < DROP1:
        return lerp(1.7, 0.6, (f - FLY1) / (DROP1 - FLY1))
    if DROP1 <= f < RAMP1:
        return 0.6
    if RAMP1 <= f < 330:
        return 1.6
    if 330 <= f < 346:
        return 2.3
    if 346 <= f < 352:
        return lerp(2.3, 0.0, (f - 346) / 6)
    return 0.0


for i, j in enumerate(jets):
    vals = []
    for f in ALL_FRAMES:
        L = jet_len(f)
        if L <= 0.02:
            vals.append((0.0, 0.0, 0.0))
        else:
            L *= 1 + 0.08 * math.sin(f * 2.1 + i * 1.7)
            vals.append((1.0, 1.0, L))
    bake(j, "scale", ALL_FRAMES, vals)


# ---------------------------------------------------------------- rival slabs

SLABS = [  # name, x along the road, size (ChatGPT is the big "Semi")
    ("chatgpt", 9.6, 4.0, srgb(250, 250, 250)),
    ("claude", 14.6, 3.2, srgb(244, 240, 230)),
    ("gemini", 19.6, 3.2, srgb(250, 250, 250)),
    ("deepseek", 24.6, 3.2, srgb(250, 250, 250)),
    ("grok", 29.6, 3.2, srgb(10, 10, 10)),
]
THICK = 0.8
YAW = math.radians(-50)     # faces turned toward the start / the stands
FLOAT = 0.5


def impact_frame(x):
    for f in range(FLY0, FLY1 + 1):
        if fly_x(f) >= x - 0.4:
            return f
    return FLY1


chip_me = box_mesh("Chip", (1, 1, 1))
puff_tex = bpy.data.textures.new("PuffClouds", 'CLOUDS')
puff_tex.noise_scale = 0.5
bm = bmesh.new()
bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
puff_me = bpy.data.meshes.new("Puff")
bm.to_mesh(puff_me)
bm.free()
for p in puff_me.polygons:
    p.use_smooth = True
puffs_root = empty("PUFFS", size=0.1)
debris_root = empty("DEBRIS", size=0.1)
puff_count = [0]


def spawn_puff(pos, f0, size, life, drift, mat):
    puff_count[0] += 1
    ob = link(f"Puff_{puff_count[0]:03d}", puff_me, None, puffs_root)
    ob.material_slots and None
    ob.data = puff_me
    ob.active_material = mat
    ob.material_slots[0].link = 'OBJECT'
    ob.material_slots[0].material = mat
    ob.rotation_euler = (rnd.uniform(0, 6.28), rnd.uniform(0, 6.28), rnd.uniform(0, 6.28))
    d = ob.modifiers.new("Puffy", 'DISPLACE')
    d.texture = puff_tex
    d.strength = 0.45
    d.texture_coords = 'LOCAL'
    frames = [1, max(2, f0 - 1), f0, f0 + int(life * 0.35), f0 + int(life * 0.7), f0 + life]
    sc = [0, 0, 0.25 * size, size, size * 1.08, 0]
    loc = [pos, pos, pos, pos + drift * 0.55, pos + drift * 0.85, pos + drift]
    frames, sc, loc = zip(*sorted({f: (f, s_, l_) for f, s_, l_ in zip(frames, sc, loc)}.values()))
    bake(ob, "scale", list(frames), [(s_, s_, s_ * 0.8) for s_ in sc], 'BEZIER')
    bake(ob, "location", list(frames), [tuple(l_) for l_ in loc], 'BEZIER')


def spawn_chips(center, f0, mat, n, spread):
    for c in range(n):
        ob = link(f"Chip_{f0}_{c}", chip_me, None, debris_root)
        ob.active_material = mat
        ob.material_slots[0].link = 'OBJECT'
        ob.material_slots[0].material = mat
        sz = rnd.uniform(0.12, 0.34)
        p = center + Vector((rnd.uniform(-spread, spread), rnd.uniform(-0.4, 0.4), rnd.uniform(0, spread)))
        v = Vector((rnd.uniform(-3.5, 3.5), rnd.uniform(-4.5, 1.5), rnd.uniform(2.0, 6.0)))
        spin = Vector((rnd.uniform(-0.6, 0.6), rnd.uniform(-0.6, 0.6), rnd.uniform(-0.6, 0.6)))
        rot = Vector((0, 0, 0))
        frames, locs, rots, scs = [1, f0 - 1], [tuple(p)] * 2, [(0, 0, 0)] * 2, [(0, 0, 0)] * 2
        resting = False
        for k in range(0, 46):
            f = f0 + k
            if f > FRAMES:
                break
            if not resting:
                p = p + v / FPS
                v.z -= 9.8 / FPS
                rot = rot + spin
                if p.z <= sz * 0.25:
                    p.z = sz * 0.25
                    resting = True
            frames.append(f)
            locs.append(tuple(p))
            rots.append(tuple(rot))
            scs.append((sz, sz * 0.7, sz * 0.35))
        bake(ob, "location", frames, locs)
        bake(ob, "rotation_euler", frames, rots)
        bake(ob, "scale", frames, scs, 'CONSTANT')


impacts = {}
for idx, (name, sx, size, col) in enumerate(SLABS):
    body_mat = material(f"Slab_{name}", col, rough=0.25)
    face_mats = {st: material(f"Face_{name}_{st}", (1, 1, 1), rough=0.3, emit=0.6,
                              image=f"{name}_{st}.png") for st in ("clean", "cracked", "glitch")}
    root = empty(f"SLAB_{name}", size=size * 0.6)
    root.rotation_mode = 'QUATERNION'
    rounded_box(f"Slab_{name}_Body", (size, THICK, size), size * 0.17, body_mat, root)
    faces = {st: decal(f"Slab_{name}_Face_{st}", size * 0.86, size * 0.86, m, root,
                       (0, -THICK / 2 - 0.004 - 0.001 * k, 0))
             for k, (st, m) in enumerate(face_mats.items())}

    fi = impact_frame(sx)
    impacts[name] = fi
    land = fi + 5           # bottom edge hits the asphalt, face cracks
    flat = land + 8         # tipped over onto its back, logo up
    q_yaw = Quaternion((0, 0, 1), YAW + math.radians(rnd.uniform(-6, 6)))
    z_float = FLOAT + size / 2
    # tip over the back bottom edge, which stays put on the asphalt
    edge = Vector((0, THICK / 2, -size / 2))
    pivot = Vector((sx, 0, 0)) + q_yaw @ Vector((0, THICK / 2, 0))
    BETA0, BETA1 = math.radians(-8), math.radians(-90)

    def tipped(beta):
        q = q_yaw @ Quaternion((1, 0, 0), beta)
        return pivot - q @ edge, q

    c_land, q_land = tipped(BETA0)
    q_flat = tipped(BETA1)[1]
    phase = rnd.uniform(0, 6.28)
    twitch = set()
    f = 296 + rnd.randint(0, 8)
    while f < FRAMES:
        for k in range(rnd.randint(2, 4)):
            twitch.add(f + k)
        f += rnd.randint(7, 15)
    pos, rot = [], []
    for f in ALL_FRAMES:
        if f < fi:
            pos.append(Vector((sx, 0, z_float + 0.07 * math.sin(f / FPS * 3.9 + phase))))
            rot.append(q_yaw)
        elif f < land:
            u = (f - fi) / (land - fi)
            pos.append(Vector((sx, 0, z_float)).lerp(c_land, u * u))
            rot.append(q_yaw.slerp(q_land, u * u))
        elif f < flat:
            u = (f - land) / (flat - land)
            c, q = tipped(lerp(BETA0, BETA1, u * u))
            pos.append(c)
            rot.append(q)
        else:
            k = f - flat
            c, q = tipped(BETA1 + (math.radians(7) * math.sin(math.pi * k / 5) if k < 5 else 0))
            if f in twitch:
                c = c + Vector((0, 0, rnd.uniform(0.0, 0.06)))
                q = q @ Quaternion((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)),
                                   math.radians(rnd.uniform(1.5, 4)))
            pos.append(c)
            rot.append(q)
    bake(root, "location", ALL_FRAMES, [tuple(p) for p in pos])
    bake(root, "rotation_quaternion", ALL_FRAMES, [tuple(q) for q in fix_signs(rot)])

    # face states: clean -> cracked on landing -> glitch flashes from 12 s
    state = []
    for f in ALL_FRAMES:
        if f < land:
            state.append("clean")
        elif f in twitch:
            state.append("glitch")
        else:
            state.append("cracked")
    for st, ob in faces.items():
        bake(ob, "scale", ALL_FRAMES, [(1, 1, 1) if s_ == st else (0, 0, 0) for s_ in state], 'CONSTANT')

    hit = Vector((sx, 0, z_float + size * 0.3))
    spawn_chips(hit, land - 1, body_mat, 14, size * 0.35)
    slam = tipped(BETA1)[0]
    for k in range(3):
        spawn_puff(slam + Vector((rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5), -0.1)), flat + k,
                   rnd.uniform(0.5, 0.8), 26, Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), 0.5)), M["vapor"])
    for k in range(4):
        spawn_puff(Vector((sx + rnd.uniform(-1, 1), rnd.uniform(-0.6, 0.8), 0.6)), land - 2 + k,
                   rnd.uniform(0.8, 1.3), 34, Vector((rnd.uniform(-1.2, 1.2), rnd.uniform(-0.8, 0.8), 1.6)),
                   M["smoke"])

# compressed-air bursts at the start, lift-off cloud, landing cloud
for p in PUFFS:
    for (nx, ny) in NOZ:
        spawn_puff(Vector((nx * 1.2, ny * 1.2, 0.25)), p + 1, rnd.uniform(0.45, 0.7), 26,
                   Vector((nx * 1.1, ny * 1.1, 0.25)), M["vapor"])
for k in range(7):
    a = k / 7 * 2 * math.pi
    spawn_puff(Vector((math.cos(a) * 1.5, math.sin(a) * 1.5, 0.35)), LIFT0 + k % 3, rnd.uniform(0.8, 1.2), 40,
               Vector((math.cos(a) * 2.2, math.sin(a) * 2.2, 0.4)), M["vapor"])
for k in range(8):
    a = k / 8 * 2 * math.pi + 0.3
    spawn_puff(Vector((2.5 + math.cos(a) * 1.5, math.sin(a) * 1.5, 0.3)), 334 + k % 4, rnd.uniform(0.6, 0.9),
               60, Vector((math.cos(a) * 1.8, math.sin(a) * 1.8, 0.4)), M["vapor"])


# ---------------------------------------------------------------- cameras

def look_euler(loc, target, up_hint):
    f = (Vector(target) - Vector(loc)).normalized()
    right = f.cross(Vector(up_hint)).normalized()
    up = right.cross(f)
    return Matrix((right, up, -f)).transposed().to_euler()


def make_cam(name, loc, target=None, lens=26, shake=0.006, loc_end=None, target_end=None,
             f0=1, f1=FRAMES, track=None, euler=None):
    rig = empty(f"RIG_{name}", loc, size=0.5)
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.sensor_height = 24
    cam_data.clip_end = 1000
    cam = link(name, cam_data, None, rig)
    if euler is not None:
        rig.rotation_euler = euler
    else:
        tgt = track
        if tgt is None:
            tgt = empty(f"TGT_{name}", target, size=0.3)
            if target_end is not None:
                bake(tgt, "location", [f0, f1], [tuple(target), tuple(target_end)], 'BEZIER')
        c = rig.constraints.new('TRACK_TO')
        c.target = tgt
        c.track_axis = 'TRACK_NEGATIVE_Z'
        c.up_axis = 'UP_Y'
    if loc_end is not None:
        bake(rig, "location", [f0, f1], [tuple(loc), tuple(loc_end)], 'BEZIER')
    if shake:
        cam.keyframe_insert("rotation_euler", frame=1)
        for k, fc in enumerate(_fcurves(cam)):
            if fc.data_path != "rotation_euler":
                continue
            mod = fc.modifiers.new('NOISE')
            mod.scale = 9.0
            mod.strength = shake * (0.5 if k == 2 else 1.0)
            mod.phase = rnd.uniform(0, 100)
    return cam


SHOTS = [  # name, first frame, camera kwargs
    ("SH01_Start", 1, dict(loc=(-6.0, -10.0, 3.0), target=(6.1, 0.5, 1.2), lens=14,
                           loc_end=(-5.4, -9.6, 2.9), f0=1, f1=48)),
    ("SH02_Puffs_LowFront", 49, dict(loc=(-6.5, -4.2, 2.4), target=(0, 0, 0.8), lens=24,
                                    target_end=(0.8, 0, 3.8), f0=49, f1=74)),
    ("SH03_Lineup_FirstHit", 75, dict(loc=(0.5, -10.5, 1.3), target=(14, 0, 3.4), lens=22,
                                     loc_end=(1.0, -10.3, 1.3), f0=75, f1=122)),
    ("SH04_KeyMoment", 123, dict(loc=(2.0, -6.5, 1.0), target=(22, 2, 6.5), lens=18,
                                target_end=(26, 2, 8.0), f0=123, f1=186)),
    ("SH05_OverTheRow", 187, dict(loc=(21.0, -8.0, 6.0), target=(35, 0, 4.0), lens=20,
                                  loc_end=(22.5, -7.6, 6.2), target_end=(38, 0, 8.0), f0=187, f1=216)),
    ("SH06_Loop_Wide", 217, dict(loc=(33, -50, 10), target=(34, 0, 10.5), lens=26, shake=0.004)),
    ("SH07_Roll_LowUp", 291, dict(loc=(9.5, -9.5, 0.9), lens=30, track=genora)),
    ("SH08_Landing", 317, dict(loc=(-7.5, -6.5, 2.7), target=(3, 0.5, 1.8), lens=24,
                              loc_end=(-6.9, -6.1, 2.6), f0=317, f1=360)),
]
cams = {}
for name, f0, kw in SHOTS:
    cams[name] = make_cam(name, **kw)
    m = scene.timeline_markers.new(name, frame=f0)
    m.camera = cams[name]
scene.camera = cams["SH01_Start"]

# one continuous handheld take from the stands
ema = gpos[0].copy()
tgt_vals = []
CENTER = Vector((18, 0, 3))
for i, f in enumerate(ALL_FRAMES):
    ema = ema.lerp(gpos[i], 0.12)
    tgt_vals.append(tuple(ema.lerp(CENTER, 0.22)))
ot_tgt = empty("TGT_OneTake", tgt_vals[0], size=0.3)
bake(ot_tgt, "location", ALL_FRAMES, tgt_vals)
cam_ot = make_cam("CAM_OneTake_Stands", (8.0, -21.0, 6.5), lens=24, shake=0.012, track=ot_tgt)


# ---------------------------------------------------------------- render setup

def setup_render(sc):
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
    sc.render.resolution_percentage = 100
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'TEXTURE'
    sh.show_shadows = True
    sh.shadow_intensity = 0.55
    sh.show_cavity = True
    sh.cavity_type = 'BOTH'
    sh.show_object_outline = True
    sh.object_outline_color = (0.0, 0.0, 0.0)
    sc.display.render_aa = '8'
    sc.display.light_direction = (0.35, -0.45, 0.82)
    sc.view_settings.view_transform = 'Standard'
    r = sc.render
    r.use_stamp = True
    r.use_stamp_date = r.use_stamp_render_time = r.use_stamp_memory = False
    r.use_stamp_time = r.use_stamp_hostname = r.use_stamp_filename = False
    r.use_stamp_lens = True
    r.use_stamp_camera = r.use_stamp_frame = r.use_stamp_note = True
    r.use_stamp_scene = False
    r.stamp_note_text = "GENORA - grey boardomatic v2"
    r.stamp_font_size = 26
    r.stamp_background = (0, 0, 0, 0.6)
    if sc.world is None:
        sc.world = bpy.data.worlds.new("NightGrey")
    sc.world.color = srgb(34, 38, 48)


setup_render(scene)
one = bpy.data.scenes.new("Boardomatic_OneTake")
one.collection.children.link(SET)
setup_render(one)
one.world = scene.world
one.camera = cam_ot

if GREY:
    CLAY = [  # material name prefix -> (grey level, alpha)
        ("Ground", 0.30, 1), ("Road", 0.20, 1), ("WhiteLine", 0.50, 1), ("RedLED", 0.50, 1),
        ("Truss", 0.45, 1), ("Mast", 0.40, 1), ("FloodLamp", 0.85, 1),
        ("GenoraBody", 0.16, 1), ("GenoraTop", 0.16, 1), ("Chrome", 0.45, 1),
        ("Jet", 0.85, 0.55), ("Vapor", 0.78, 0.5), ("ImpactSmoke", 0.78, 0.5),
        ("Slab_", 0.62, 1), ("Face_", 0.62, 1),
    ]
    for m in bpy.data.materials:
        for prefix, g, a in CLAY:
            if m.name.startswith(prefix):
                m.diffuse_color = (g, g, g, a)
                break
    for sc in (scene, one):
        sc.display.shading.color_type = 'MATERIAL'
        sc.render.use_stamp = False
    scene.world.color = (0.12, 0.12, 0.12)

readme = bpy.data.texts.new("README_RU")
readme.write(
    "GENORA - серый бордоматик v1\n\n"
    "Сцены (переключатель сцен вверху справа):\n"
    "  Boardomatic_Cuts    - монтаж из 8 планов, камеры переключают маркеры SH01..SH08\n"
    "  Boardomatic_OneTake - один непрерывный план с трибуны (как телефонная съёмка)\n\n"
    "Тайминг 24 fps, 360 кадров (15 с):\n"
    "  0-3 с   пшики сжатого воздуха, взлёт\n"
    f"  3-8 с   пролёт над рядом, удары по плитам: {impacts}\n"
    "  8-12 с  посадка на дорогу, мёртвая петля вверх ногами, вылет с открытого конца\n"
    "  12-15 с переворот на 180, пролёт над разбитыми плитами (глитч), посадка в дым\n\n"
    "Объекты:\n"
    "  GENORA            корень машины (анимация), Genora_Jet_* - струи (scale Z = длина)\n"
    "  SLAB_<name>       иконки; Face_clean / cracked / glitch переключаются scale 0/1\n"
    "  TRACK             дорога, рампа, ферма, мачты\n"
    "  PUFFS / DEBRIS    дым и осколки\n\n"
    "Лого - оригинальные иконки приложений (icons/), текстуры собраны в textures/.\n"
    "Рендер: Workbench (серый превиз). Для красивого света переключите на EEVEE -\n"
    "у материалов уже есть эмиссия (LED, прожекторы, струи, лого).\n"
)

bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print("IMPACTS", impacts)
print("SAVED", OUT)
