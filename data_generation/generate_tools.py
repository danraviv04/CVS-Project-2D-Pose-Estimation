#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------
# Synthetic OR tools + hands: stable, deterministic layout
# -------------------------------

import blenderproc as bproc
from blenderproc.python.camera import CameraUtility

import bpy
import numpy as np
from numpy import pi
import argparse, random, os, json, shutil, glob
from math import radians
from math import atan2
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
from mathutils.bvhtree import BVHTree
from PIL import Image

# =========================
# Small node-graph helpers
# =========================

def _get_input(node, *names):
    for n in names:
        s = node.inputs.get(n)
        if s is not None:
            return s
    return None

def _set_input(node, value, *names):
    s = _get_input(node, *names)
    if s is not None:
        s.default_value = value

# =========================
# Shading utilities
# =========================

def _shade_smooth(obj_bpy, angle_deg=60.0):
    """Robust smooth shading across Blender 2.9/3.x/4.x."""
    if not obj_bpy or obj_bpy.type != 'MESH':
        return
    me = obj_bpy.data
    try:
        obj_bpy.select_set(True)
        bpy.context.view_layer.objects.active = obj_bpy
        bpy.ops.object.shade_smooth()
    except Exception:
        pass

    if hasattr(me, "polygons"):
        for p in me.polygons:
            try:
                p.use_smooth = True
            except Exception:
                break

    # Auto smooth (<=3.x) or fallback modifiers
    from math import radians as _rad
    if hasattr(me, "use_auto_smooth"):
        try:
            me.use_auto_smooth = True
        except Exception:
            pass
        if hasattr(me, "auto_smooth_angle"):
            try:
                me.auto_smooth_angle = _rad(float(angle_deg))
            except Exception:
                pass
    else:
        if not any(m.type == 'WEIGHTED_NORMAL' for m in obj_bpy.modifiers):
            try:
                m = obj_bpy.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
                m.keep_sharp = True
                m.mode = 'FACE_AREA'
                m.weight = 50
            except Exception:
                pass
        if hasattr(bpy.types, "EdgeSplitModifier") and not any(m.type == 'EDGE_SPLIT' for m in obj_bpy.modifiers):
            try:
                es = obj_bpy.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
                es.use_edge_angle = True
                es.use_edge_sharp = True
                es.split_angle = _rad(float(angle_deg))
            except Exception:
                pass
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

def ensure_glove_material(name="GloveSSS", tint=(1.0, 1.0, 1.0)):
    """Soft SSS nitrile glove with tiny micro-normal breakup; reuses on subsequent calls, re-jitters a bit."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out  = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

        _set_input(bsdf, (0.90, 0.90, 0.92, 1.0), "Base Color")
        _set_input(bsdf, np.random.uniform(0.28, 0.42), "Subsurface", "Subsurface Weight")
        _set_input(bsdf, (1.2, 0.6, 0.45), "Subsurface Radius")
        _set_input(bsdf, 0.35, "Specular", "Specular IOR Level")
        _set_input(bsdf, np.random.uniform(0.38, 0.55), "Roughness")
        _set_input(bsdf, 0.15, "Sheen")
        _set_input(bsdf, 0.08, "Clearcoat")
        _set_input(bsdf, 0.05, "Clearcoat Roughness")

        texcoord = nt.nodes.new("ShaderNodeTexCoord")
        mapping  = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])

        noise_hi = nt.nodes.new("ShaderNodeTexNoise")
        noise_hi.inputs["Scale"].default_value = 300.0
        noise_hi.inputs["Detail"].default_value = 2.0
        noise_hi.inputs["Roughness"].default_value = 0.45
        nt.links.new(mapping.outputs["Vector"], noise_hi.inputs["Vector"])

        musgrave_ok = hasattr(bpy.types, "ShaderNodeTexMusgrave")
        voronoi_ok  = hasattr(bpy.types, "ShaderNodeTexVoronoi")

        if musgrave_ok:
            tex2 = nt.nodes.new("ShaderNodeTexMusgrave")
            try:
                tex2.musgrave_type = 'RIDGED_MULTI_FRACTAL'
            except Exception:
                pass
            tex2.inputs["Scale"].default_value = 120.0
            tex2.inputs["Detail"].default_value = 4.0
            nt.links.new(mapping.outputs["Vector"], tex2.inputs["Vector"])
            tex2_out = tex2.outputs.get("Fac") or tex2.outputs[0]
        elif voronoi_ok:
            tex2 = nt.nodes.new("ShaderNodeTexVoronoi")
            tex2.inputs["Scale"].default_value = 120.0
            nt.links.new(mapping.outputs["Vector"], tex2.inputs["Vector"])
            tex2_out = tex2.outputs.get("Distance") or tex2.outputs[0]
        else:
            tex2 = nt.nodes.new("ShaderNodeTexNoise")
            tex2.inputs["Scale"].default_value = 120.0
            tex2.inputs["Detail"].default_value = 4.0
            tex2.inputs["Roughness"].default_value = 0.5
            nt.links.new(mapping.outputs["Vector"], tex2.inputs["Vector"])
            tex2_out = tex2.outputs.get("Fac") or tex2.outputs[0]

        mixn = nt.nodes.new("ShaderNodeMixRGB")
        mixn.blend_type = 'ADD'
        mixn.inputs["Fac"].default_value = 0.5
        nt.links.new(noise_hi.outputs["Fac"], mixn.inputs[1])
        nt.links.new(tex2_out,                mixn.inputs[2])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.25
        nt.links.new(mixn.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        geom  = nt.nodes.new("ShaderNodeNewGeometry")
        cramp = nt.nodes.new("ShaderNodeValToRGB")
        cramp.color_ramp.elements[0].position = 0.35
        cramp.color_ramp.elements[1].position = 0.80
        if "Pointiness" in geom.outputs:
            nt.links.new(geom.outputs["Pointiness"], cramp.inputs["Fac"])
        mult = nt.nodes.new("ShaderNodeMixRGB")
        mult.blend_type = 'MULTIPLY'
        mult.inputs["Fac"].default_value = 0.35
        mult.inputs["Color2"].default_value = (0.85, 0.85, 0.85, 1.0)
        nt.links.new(cramp.outputs["Color"], mult.inputs["Color1"])

        tintnode = nt.nodes.new("ShaderNodeRGB")
        tintnode.outputs[0].default_value = (*tint, 1.0)
        mix_tint = nt.nodes.new("ShaderNodeMixRGB")
        mix_tint.blend_type = 'MIX'
        mix_tint.inputs["Fac"].default_value = 0.15
        nt.links.new(tintnode.outputs[0],   mix_tint.inputs["Color2"])
        nt.links.new(mult.outputs["Color"], mix_tint.inputs["Color1"])
        nt.links.new(mix_tint.outputs["Color"], bsdf.inputs["Base Color"])

        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    else:
        bsdf = next((n for n in mat.node_tree.nodes if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)), None)
        if bsdf:
            _set_input(bsdf, np.random.uniform(0.28, 0.42), "Subsurface", "Subsurface Weight")
            _set_input(bsdf, np.random.uniform(0.38, 0.55), "Roughness")
        for n in mat.node_tree.nodes:
            if isinstance(n, bpy.types.ShaderNodeRGB):
                n.outputs[0].default_value = (*tint, 1.0)
                break
    return mat

def apply_glove_shader(hand_bproc_obj, tint=(1,1,1)):
    mat = ensure_glove_material(tint=tint)
    ob = hand_bproc_obj.blender_obj
    if len(ob.data.materials) == 0:
        ob.data.materials.append(mat)
    else:
        for i in range(len(ob.data.materials)):
            ob.data.materials[i] = mat
    _shade_smooth(ob)

def ensure_surgical_metal(name="SurgicalSteel"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out  = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

        _set_input(bsdf, (0.64, 0.66, 0.70, 1.0), "Base Color")
        _set_input(bsdf, 1.0, "Metallic")
        _set_input(bsdf, 0.85, "Specular", "Specular IOR Level")
        _set_input(bsdf, float(np.random.uniform(0.04, 0.09)), "Roughness")
        _set_input(bsdf, 0.65, "Anisotropic", "Anisotropy")
        _set_input(bsdf, float(np.random.uniform(-0.15, 0.15)), "Anisotropic Rotation", "Anisotropy Rotation")
        _set_input(bsdf, 0.20, "Clearcoat")
        _set_input(bsdf, 0.03, "Clearcoat Roughness")

        texcoord = nt.nodes.new("ShaderNodeTexCoord")
        mapping  = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
        try:
            mapping.inputs["Scale"].default_value = (6.0, 1.0, 1.0)
        except Exception:
            try:
                v = mapping.inputs["Scale"].default_value
                v[0], v[1], v[2] = 6.0, 1.0, 1.0
            except Exception:
                pass

        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 260.0
        noise.inputs["Detail"].default_value = 2.0
        noise.inputs["Roughness"].default_value = 0.35
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        try:
            wave = nt.nodes.new("ShaderNodeTexWave")
            wave.wave_type = 'BANDS'
            wave.inputs["Scale"].default_value = 700.0
            wave.inputs["Distortion"].default_value = 5.0
            wave.inputs["Detail"].default_value = 0.0
            nt.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

            mix = nt.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = 'ADD'
            mix.inputs["Fac"].default_value = 0.5
            nt.links.new(noise.outputs["Fac"], mix.inputs[1])
            nt.links.new(wave.outputs["Color"], mix.inputs[2])
            height_src = mix.outputs["Color"]
        except Exception:
            height_src = noise.outputs["Fac"]

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        nt.links.new(height_src, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        try:
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Clearcoat Normal"])
        except Exception:
            pass

        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    else:
        bsdf = next((n for n in mat.node_tree.nodes if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)), None)
        if bsdf:
            _set_input(bsdf, float(np.random.uniform(0.04, 0.09)), "Roughness")
            _set_input(bsdf, 0.65, "Anisotropic", "Anisotropy")
            _set_input(bsdf, float(np.random.uniform(-0.15, 0.15)), "Anisotropic Rotation", "Anisotropy Rotation")
    return mat

def apply_surgical_metal(tool_bproc_obj):
    mat = ensure_surgical_metal()
    ob = tool_bproc_obj.blender_obj
    if len(ob.data.materials) == 0:
        ob.data.materials.append(mat)
    else:
        for i in range(len(ob.data.materials)):
            ob.data.materials[i] = mat
    _shade_smooth(ob)

# =========================
# Wrist stub (short forearm)
# =========================

def _basis_from_z(z_world, up_hint=Vector((0,0,1))):
    z = z_world.normalized()
    x = up_hint.cross(z)
    if x.length < 1e-6:
        x = Vector((1,0,0)).cross(z)
    x = x.normalized()
    y = z.cross(x).normalized()
    return Matrix(((x.x,y.x,z.x),
                   (x.y,y.y,z.y),
                   (x.z,y.z,z.z)))

def add_wrist_stub(hand_bproc_obj, mat=None, length_mult=(0.035, 0.06), radius_mult=0.95):
    hand = hand_bproc_obj
    H = np.asarray(hand.get_local2world_mat(), dtype=float)
    fL = HAND_FWD_LOCAL.normalized()

    Vloc = np.array([v.co[:] for v in hand.get_mesh().vertices], dtype=float)
    proj = Vloc @ np.array([fL.x, fL.y, fL.z])
    k = proj.min() + 0.003
    sel = Vloc[proj <= k + 1e-4]
    if sel.shape[0] < 6:
        return None

    def _w(p):
        w4 = H @ np.array([p[0], p[1], p[2], 1.0])
        return Vector((float(w4[0]), float(w4[1]), float(w4[2])))

    ring_w = np.array([_w(p) for p in sel])
    center = Vector(ring_w.mean(0))
    fwd_w = (hand.blender_obj.matrix_world.to_3x3() @ fL).normalized()
    U = np.eye(3) - np.outer(np.array([fwd_w.x,fwd_w.y,fwd_w.z]),
                             np.array([fwd_w.x,fwd_w.y,fwd_w.z]))
    q = (ring_w - ring_w.mean(0)) @ U.T
    rad = float(np.sqrt((q**2).sum(1).mean()))
    length = np.random.uniform(*length_mult)

    bpy.ops.mesh.primitive_cylinder_add(radius=rad*radius_mult, depth=length)
    cyl = bpy.context.object
    cyl.name = f"{hand.get_name()}_WristStub"
    R = _basis_from_z(fwd_w).to_4x4()
    T = Matrix.Translation(center + fwd_w * (length*0.5))
    cyl.matrix_world = T @ R
    if mat is None:
        mat = ensure_glove_material()
    if len(cyl.data.materials) == 0:
        cyl.data.materials.append(mat)
    else:
        cyl.data.materials[0] = mat
    _shade_smooth(cyl)
    cyl.parent = hand.blender_obj
    cyl.matrix_parent_inverse = hand.blender_obj.matrix_world.inverted()
    return cyl

# =========================
# CONFIG (your knobs)
# =========================

NUM_IMAGES = 3
BASE_PATH = "/datashare/project"
TOOLS_DEFAULT_FOLDER = os.path.join(BASE_PATH, "surgical_tools_models")
CAMERA_PARAMS_DEFAULT = os.path.join(BASE_PATH, "camera.json")
RIGHT_HAND_FILE = "hands/hand_1.obj"
OUTPUT_DIR_DEFAULT = "output"

# Hand local basis (RIGHT hand asset, palm-down)
HAND_RIGHT_LOCAL = Vector(( 1, 0, 0))  # thumb axis
HAND_FWD_LOCAL   = Vector(( 0, 1, 0))  # fingers axis
HAND_PALM_LOCAL  = Vector(( 0, 0,-1))  # palm normal (down)

# Size & scaling
SCALE_RATIO   = {"NH": 0.30, "T": 0.32}  # hand-length : tool-length
BASE_HAND_SCALE = 0.6
SCALE_MIN, SCALE_MAX = 0.1, 1.8
HAND_SPAN_PERCENTILES = (10, 90)

# Palm anchor along shaft
ANCHOR_FRAC   = {"NH": 0.12, "T": 0.20}
ANCHOR_JITTER = {"NH": 0.00, "T": 0.02}
ANCHOR_DRIFT  = {"NH": 0.00, "T": -0.01}
ANCHOR_CLAMP  = (0.02, 0.98)

# Orientation knobs
FINGER_DIR     = {"NH": +1, "T": +1}  # +1 → fingers → tip
SIDE_PREF      = {"NH": +1, "T": +1}  # preferred side; auto flips for left hand
ROLL_MAX_DEG   = 16
ROLL_BIAS_RIGHT= 6

# Clearances / visibility (scale by end thickness)
UP_SCALE     = {"NH": 0.30, "T": 0.25}
SIDE_SCALE   = {"NH": 0.35, "T": 0.25}
MIN_UP_MM    = {"NH": 0.0015, "T": 0.0006}
MIN_SIDE_MM  = {"NH": 0.0010, "T": 0.0005}
CONTACT_MM   = 0.0008
FRONT_MM     = 0.0
MAX_RETRIES_VISIBLE = 8
MARGIN = 0.02

# Base/tip hints (stable + cached)
T_AXIS_HINT   = Vector((0, 1, 0))
T_BASE_SIGN   = -1      # tweak once if a family is reversed
NH_BASE_SIGN  = +1      # exists now; used only if thickness tie
THICK_TOL     = 1.03

# Back-compat (do not touch rest of code)
RATIO_TWEEZERS, RATIO_NEEDLEHOLD = SCALE_RATIO["T"], SCALE_RATIO["NH"]
RATIO_JITTER = 0.0
ALONG_DRIFT  = ANCHOR_DRIFT
SIDE_DIR     = SIDE_PREF
MIN_UP_MM_KIND, MIN_SIDE_MM_KIND = MIN_UP_MM, MIN_SIDE_MM

# =========================
# Geometric helpers
# =========================

_BVH_CACHE = {}

def _bvh_from_object(obj):
    key = obj.blender_obj.as_pointer()
    if key in _BVH_CACHE:
        return _BVH_CACHE[key]
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = obj.blender_obj.evaluated_get(deps)
    me = ob_eval.to_mesh()
    me.transform(ob_eval.matrix_world)
    me.calc_loop_triangles()
    verts = [v.co.copy() for v in me.vertices]
    tris  = [tuple(loop.vertices) for loop in me.loop_triangles]
    tree  = BVHTree.FromPolygons(verts, tris)
    ob_eval.to_mesh_clear()
    _BVH_CACHE[key] = tree
    return tree

def _hand_world_centroid(hand):
    H = np.asarray(hand.get_local2world_mat(), dtype=float)
    V = np.array([v.co[:] for v in hand.get_mesh().vertices], dtype=float)
    m = V.mean(0)
    w4 = H @ np.array([m[0], m[1], m[2], 1.0])
    return Vector((float(w4[0]), float(w4[1]), float(w4[2])))

def _nudge_to_surface(hand, tool, up, side, anchor, side_mm, lift_mm):
    tree = _bvh_from_object(tool)
    starts = [
        anchor + side*side_mm + up*(lift_mm + 0.004),
        anchor + side*side_mm + up*(lift_mm + 0.002),
        anchor + side*side_mm + up*(lift_mm + 0.001),
    ]
    for s in starts:
        hit = tree.ray_cast(s, -up, 0.05)
        if hit[0] is not None:
            co, n, _, _ = hit
            if n.dot(up) < 0: n = -n
            H = hand.blender_obj.matrix_world
            c = _hand_world_centroid(hand)
            target = co + n * CONTACT_MM
            hand.blender_obj.matrix_world = Matrix.Translation(target - c) @ H
            bpy.context.view_layer.update()
            return True
    return False

def _stabilize_basis_with_local_axes(obj, fwd, up):
    """Sign-stabilize fwd/up using object’s own local axes (prevents random flips)."""
    Rw = obj.blender_obj.matrix_world.to_3x3()
    axes_local = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]
    axes_world = [ (Rw @ a).normalized() for a in axes_local ]

    i_f = int(np.argmax([abs(fwd.dot(ax)) for ax in axes_world]))
    if fwd.dot(axes_world[i_f]) < 0: fwd = -fwd

    i_u = int(np.argmax([abs(up.dot(ax)) for ax in axes_world]))
    if up.dot(axes_world[i_u]) < 0: up = -up

    side = fwd.cross(up).normalized()
    up   = side.cross(fwd).normalized()
    return fwd.normalized(), up.normalized(), side.normalized()

def _project_point(pt_w, scene, cam, W, H):
    p = world_to_camera_view(scene, cam, Vector(pt_w))
    if not (MARGIN <= p.x <= 1.0 - MARGIN and MARGIN <= p.y <= 1.0 - MARGIN and p.z >= 0.0):
        return [0, 0, 0]
    x = int(np.clip(p.x, 0, 1) * W)
    y = int((1 - np.clip(p.y, 0, 1)) * H)
    return [x, y, 2]

def _closest_point_on_line(vec, base, shaft_u):
    t = np.dot(vec - base, shaft_u)
    return base + t * shaft_u

def _ring_points_world(Vw, base_w, shaft, side_vec, length):
    base = np.array([base_w.x, base_w.y, base_w.z], dtype=np.float32)
    shaft_u = np.array([shaft.x, shaft.y, shaft.z], dtype=np.float32)
    shaft_u /= (np.linalg.norm(shaft_u) + 1e-9)

    # keep only a slab near the base (first ~30–35% of length)
    t_along = (Vw - base) @ shaft_u
    slab = (t_along >= 0) & (t_along <= 0.35 * length)
    if not np.any(slab):
        slab = t_along <= 0.50 * length  # fallback

    P = Vw[slab]
    if len(P) < 10:  # safety
        P = Vw

    # split by world-space side
    s = side_vec / (np.linalg.norm(side_vec) + 1e-9)
    side_score = (P - P.mean(0)) @ s
    L = P[side_score < 0]
    R = P[side_score >= 0]

    def farthest_to_shaft(points):
        if len(points) == 0:
            return None
        proj = np.array([_closest_point_on_line(p, base, shaft_u) for p in points])
        d2 = ((points - proj) ** 2).sum(1)
        k = max(5, int(0.1 * len(points)))
        idx = np.argsort(d2)[-k:]
        return points[idx].mean(0).astype(np.float32, copy=False)

    left  = farthest_to_shaft(L)
    right = farthest_to_shaft(R)

    # fallback to the pooled slab if a side is empty
    if left is None:
        left = farthest_to_shaft(P)
    if right is None:
        right = farthest_to_shaft(P)

    # final hard fallback: use base if everything failed
    if left is None:
        left = base.copy()
    if right is None:
        right = base.copy()

    return left, right

def _largest_mesh(objects):
    def bbox_diag(o):
        M = np.asarray(o.get_local2world_mat(), dtype=float)
        bb = [M @ np.array([c[0], c[1], c[2], 1.0]) for c in o.get_bound_box()]
        bb = np.array(bb)[:, :3]
        return float(np.linalg.norm(bb.max(0) - bb.min(0)))
    return max(objects, key=bbox_diag)

def _image_stats(img_path):
    im = Image.open(img_path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    L = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    p5, p95 = np.percentile(L, [5, 95])
    mask = (L >= p5) & (L <= p95)
    if mask.sum() < 10:
        return float(L.mean()), arr.mean(0)
    return float(L[mask].mean()), arr[mask].mean(0)

def setup_natural_hdr_lighting(hdr_files, bg_image_path=None):
    hdr = random.choice(hdr_files)
    print(f"🌅 HDRI: {hdr}")

    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping   = nt.nodes.new("ShaderNodeMapping")
    mapping.name = "HDRI_MAP"
    env_tex   = nt.nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(hdr, check_existing=True)
    try:
        env_tex.image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass

    mapping.inputs["Rotation"].default_value[2] = random.uniform(0.0, 2*np.pi)

    bg   = nt.nodes.new("ShaderNodeBackground")
    mult = nt.nodes.new("ShaderNodeMath"); mult.operation = 'MULTIPLY'
    out  = nt.nodes.new("ShaderNodeOutputWorld")

    nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    nt.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], mult.inputs[0])
    nt.links.new(mult.outputs["Value"], out.inputs["Surface"])

    scene = bpy.context.scene
    cam = scene.camera
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = random.choice(["None", "Low Contrast", "Medium Contrast"])
    scene.view_settings.exposure = 0.0
    if hasattr(scene, "cycles"):
        scene.cycles.sample_clamp_direct   = 2.0
        scene.cycles.sample_clamp_indirect = 1.0

    env_mult = 1.0
    key_mult = 1.0
    tint = np.array([1.0, 1.0, 1.0], dtype=float)
    if bg_image_path and os.path.exists(bg_image_path):
        lum, tint = _image_stats(bg_image_path)
        env_mult = float(np.interp(lum, [0.15, 0.70], [1.4, 0.7]))
        key_mult = float(np.interp(lum, [0.15, 0.70], [900, 350]))

    bg.inputs["Strength"].default_value = 1.0
    mult.inputs[1].default_value = env_mult

    # Clear lights
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)

    elev = radians(random.uniform(25, 55))
    azim = radians(random.uniform(-150, 150))

    if random.random() < 0.6:
        bpy.ops.object.light_add(type="AREA", location=(0, 0, 4.0))
        key = bpy.context.object
        key.name = "KEY_LIGHT"
        key.data.shape = "RECTANGLE"
        key.data.size = random.uniform(2.0, 3.5)
        key.data.size_y = random.uniform(0.8, 1.6)
        key.data.energy = key_mult * random.uniform(300, 550)
    else:
        bpy.ops.object.light_add(type="SUN")
        key = bpy.context.object
        key.name = "KEY_LIGHT"
        key.data.angle = radians(random.uniform(2.0, 6.0))
        key.data.energy = key_mult * random.uniform(2.0, 3.5)

    key.rotation_euler = (elev, 0.0, azim)
    mix = 0.35
    key.data.color = tuple((1.0 - mix) + mix * float(c) for c in tint)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3.0))
    fill = bpy.context.object; fill.name = "FILL_LIGHT"
    fill.data.shape = "SQUARE"
    fill.data.size = 2.2
    fill.data.energy = key.data.energy * 0.08
    fill.rotation_euler = (radians(80), 0, azim + radians(180))

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4.5))
    rim = bpy.context.object; rim.name = "RIM_LIGHT"
    rim.data.shape = "SQUARE"
    rim.data.size = 1.6
    rim.data.energy = key.data.energy * 0.06
    rim.rotation_euler = (radians(100), 0, azim + radians(25))

    print("✅ HDRI + natural key/fill/rim configured")
    return {"tint": tuple(map(float, tint)), "map": mapping, "key": key, "fill": fill, "rim": rim}

def _mirror_mesh_inplace_x(obj):
    bo = obj.blender_obj
    bo.data = bo.data.copy()
    bo.data.transform(Matrix.Scale(-1.0, 4, Vector((1, 0, 0))))
    try: bo.data.calc_normals()
    except: pass
    bpy.context.view_layer.update()

def _bbox_norm_from_mesh(obj, scene, cam, W, H, pad_ratio=0.02):
    M = np.asarray(obj.get_local2world_mat(), dtype=float)
    xs, ys = [], []
    for v in obj.get_mesh().vertices:
        w4 = M @ np.array([v.co.x, v.co.y, v.co.z, 1.0])
        p = world_to_camera_view(scene, cam, Vector((float(w4[0]), float(w4[1]), float(w4[2]))))
        if p.z >= 0.0:
            xs.append(np.clip(p.x, 0, 1) * W)
            ys.append((1 - np.clip(p.y, 0, 1)) * H)
    if len(xs) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx, dy = pad_ratio * (x1 - x0), pad_ratio * (y1 - y0)
    x0, x1 = max(0, x0 - dx), min(W, x1 + dx)
    y0, y1 = max(0, y0 - dy), min(H, y1 + dy)
    cx = ((x0 + x1) * 0.5) / W
    cy = ((y0 + y1) * 0.5) / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return [float(cx), float(cy), float(max(w, 1.0 / W)), float(max(h, 1.0 / H))]

def _on_screen(obj, scene, cam, margin=MARGIN):
    M = np.asarray(obj.get_local2world_mat(), dtype=float)
    mesh = obj.get_mesh()
    for v in mesh.vertices:
        w4 = M @ np.array([v.co.x, v.co.y, v.co.z, 1.0])
        p = world_to_camera_view(scene, cam, Vector((float(w4[0]), float(w4[1]), float(w4[2]))))
        if margin <= p.x <= 1.0 - margin and margin <= p.y <= 1.0 - margin and p.z >= 0.0:
            return True
    return False

def _compute_hand_scale(tool_len, hand_len, kind):
    base_ratio = RATIO_NEEDLEHOLD if kind == "NH" else RATIO_TWEEZERS
    target_len = tool_len * base_ratio
    jitter = 1.0 + np.random.uniform(-RATIO_JITTER, RATIO_JITTER)
    bias = BASE_HAND_SCALE if isinstance(BASE_HAND_SCALE, (int, float)) else 1.0
    s = (target_len / max(hand_len, 1e-6)) * jitter * bias
    return float(np.clip(s, SCALE_MIN, SCALE_MAX))

def _camera_forward(cam):
    return (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()

# =========================
# Base/tip + basis (stable)
# =========================

def _tool_axes_and_endpoints_world(obj, kind):
    """
    Returns: shaft, up, side, base_w, tip_w, length, thick_est
    - Stable & deterministic (local-space) base/tip decision with per-object caching
    - Basis signs stabilized w.r.t. local axes to avoid flips
    """
    mesh = obj.get_mesh()
    V = np.array([v.co[:] for v in mesh.vertices], dtype=float)
    mean = V.mean(0)

    # PCA (local space)
    C = np.cov((V - mean).T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]  # major, sym, minor
    major, sym, minor = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]

    # Fix eigenvector signs (local, deterministic)
    if ((V - mean) @ major).max() < -((V - mean) @ major).min(): major = -major
    if ((V - mean) @ sym  ).max() < -((V - mean) @ sym  ).min(): sym   = -sym

    proj = (V - mean) @ major
    pmin, pmax = proj.min(), proj.max()
    slab = 0.10 * (pmax - pmin) + 1e-9
    near_min = V[proj <= pmin + slab]
    near_max = V[proj >= pmax - slab]

    U = np.eye(3) - np.outer(major, major)
    def thickness(pts):
        Q = (pts - pts.mean(0)) @ U.T
        return float(np.sqrt((Q**2).sum(axis=1).mean() + 1e-12))

    t_min, t_max = thickness(near_min), thickness(near_max)

    # Local -> world
    M = np.asarray(obj.get_local2world_mat(), dtype=float)
    def _w(p3):
        w4 = M @ np.array([p3[0], p3[1], p3[2], 1.0])
        return Vector((float(w4[0]), float(w4[1]), float(w4[2])))

    # --------- Deterministic base/tip with per-object caching ----------
    bo = obj.blender_obj
    kb = f"__bt_base_local_{kind}"
    kt = f"__bt_tip_local_{kind}"

    if kb in bo and kt in bo:
        base_local = np.array(bo[kb], dtype=float)
        tip_local  = np.array(bo[kt], dtype=float)
    else:
        # Decide in local coords (frame-invariant)
        if kind == "T":
            # For tweezers: use a fixed local axis & sign for tie
            end_min_local = near_min.mean(0)
            end_max_local = near_max.mean(0)

            # Dot with *local* axis (rotation-invariant)
            axis = np.array([T_AXIS_HINT.x, T_AXIS_HINT.y, T_AXIS_HINT.z], dtype=float)
            smin = float(np.dot(end_min_local, axis))
            smax = float(np.dot(end_max_local, axis))
            pick_min = (smin <= smax) if T_BASE_SIGN < 0 else (smin >= smax)
            base_local, tip_local = (end_min_local, end_max_local) if pick_min else (end_max_local, end_min_local)

        else:  # "NH"
            # Prefer thicker end as base; tie-break by fixed local axis & sign
            if t_max / (t_min + 1e-9) > THICK_TOL:
                base_local, tip_local = near_max.mean(0), near_min.mean(0)
            elif t_min / (t_max + 1e-9) > THICK_TOL:
                base_local, tip_local = near_min.mean(0), near_max.mean(0)
            else:
                end_min_local = near_min.mean(0)
                end_max_local = near_max.mean(0)
                axis = np.array([T_AXIS_HINT.x, T_AXIS_HINT.y, T_AXIS_HINT.z], dtype=float)  # reuse hint
                smin = float(np.dot(end_min_local, axis))
                smax = float(np.dot(end_max_local, axis))
                pick_min = (smin <= smax) if NH_BASE_SIGN < 0 else (smin >= smax)
                base_local, tip_local = (end_min_local, end_max_local) if pick_min else (end_max_local, end_min_local)

        # Persist once per object (Blender ID props)
        bo[kb] = tuple(map(float, base_local))
        bo[kt] = tuple(map(float, tip_local))

    base_w, tip_w = _w(base_local), _w(tip_local)

    # Clean right-handed basis from shaft & minor
    shaft = (tip_w - base_w).normalized()
    w     = M @ np.array([minor[0], minor[1], minor[2], 0.0])
    up    = Vector((float(w[0]), float(w[1]), float(w[2])))
    side  = shaft.cross(up).normalized()
    up    = side.cross(shaft).normalized()

    # Stabilize signs with local axes
    shaft, up, side = _stabilize_basis_with_local_axes(obj, shaft, up)

    length    = max((tip_w - base_w).length, 1e-6)
    thick_est = max(t_min, t_max)
    return shaft, up, side, base_w, tip_w, length, thick_est

# =========================
# Hand alignment
# =========================

def _proj_to_plane(v: Vector, n: Vector) -> Vector:
    return (v - n * v.dot(n)).normalized()

def _roll_towards_camera(R3: Matrix, shaft: Vector, palm_local: Vector, cam,
                         max_deg=ROLL_MAX_DEG, bias_ccw_deg=0.0) -> Matrix:
    cam_fwd = _camera_forward(cam)
    u_cur   = (R3 @ palm_local).normalized()
    a = _proj_to_plane(u_cur,    shaft)
    b = _proj_to_plane(-cam_fwd, shaft)
    dot   = max(-1.0, min(1.0, a.dot(b)))
    cross = a.cross(b)
    phi   = atan2(shaft.dot(cross), dot)
    phi  += radians(bias_ccw_deg)
    phi   = float(np.clip(phi, -radians(max_deg), radians(max_deg)))
    return Matrix.Rotation(phi, 3, shaft) @ R3

NH_ROLL_FIX_DEG = 90.0

def _align_and_scale_hand_to_tool(hand, tool, kind, scene, cam):
    shaft, up0, side0, base_w, tip_w, length, thick = _tool_axes_and_endpoints_world(tool, kind)

    is_left = "left" in hand.blender_obj.name.lower()

    side = (side0 * (SIDE_DIR[kind] * (-1 if is_left else 1))).normalized()
    up   = side.cross(shaft).normalized()
    side = shaft.cross(up).normalized()

    #rL = HAND_RIGHT_LOCAL if not is_left else -HAND_RIGHT_LOCAL
    rL = HAND_RIGHT_LOCAL
    fL = HAND_FWD_LOCAL
    pL = HAND_PALM_LOCAL

    B_tool = Matrix(((side.x,  shaft.x,  up.x),
                     (side.y,  shaft.y,  up.y),
                     (side.z,  shaft.z,  up.z)))
    
    B_hand = Matrix(((rL.x,    fL.x,     pL.x),
                     (rL.y,    fL.y,     pL.y),
                     (rL.z,    fL.z,     pL.z)))
    R = B_tool @ B_hand.inverted()

    if ((R @ fL).dot(shaft) * FINGER_DIR[kind]) < 0:
        R = Matrix.Rotation(np.pi, 3, up) @ R

    bias = (ROLL_BIAS_RIGHT if not is_left else 0.0)
    # if kind == "NH":
    #     roll_fix = radians(( -1 if is_left else +1 ) * NH_ROLL_FIX_DEG)
    #     R = Matrix.Rotation(roll_fix, 3, shaft) @ R
    R = _roll_towards_camera(R, shaft, pL, cam, max_deg=ROLL_MAX_DEG, bias_ccw_deg=bias)

    loc0 = hand.blender_obj.matrix_world.to_translation().copy()
    hand.blender_obj.matrix_world = Matrix.Translation(loc0) @ R.to_4x4()
    bpy.context.view_layer.update()

    frac = float(ANCHOR_FRAC[kind]) + np.random.uniform(-ANCHOR_JITTER[kind], ANCHOR_JITTER[kind])
    frac = min(max(frac + ALONG_DRIFT[kind], ANCHOR_CLAMP[0]), ANCHOR_CLAMP[1])
    anchor = base_w + shaft * (length * frac)
    hand.set_location(anchor); bpy.context.view_layer.update()

    H = np.asarray(hand.get_local2world_mat(), dtype=float)
    sdir = np.array([shaft.x, shaft.y, shaft.z], dtype=float)
    dots = []
    for v in hand.get_mesh().vertices:
        w4 = H @ np.array([v.co.x, v.co.y, v.co.z, 1.0], dtype=float)
        dots.append(float(w4[0]*sdir[0] + w4[1]*sdir[1] + w4[2]*sdir[2]))
    lo, hi   = np.percentile(np.asarray(dots, dtype=float), HAND_SPAN_PERCENTILES)
    hand_len = max(float(hi - lo), 1e-6)
    s        = _compute_hand_scale(length, hand_len, kind)
    hand.set_scale([s, s, s]); bpy.context.view_layer.update()

    lift_mm = max(MIN_UP_MM_KIND[kind],   thick * UP_SCALE[kind])
    side_mm = max(MIN_SIDE_MM_KIND[kind], thick * SIDE_SCALE[kind])
    cam_fwd = _camera_forward(cam)

    for k in range(MAX_RETRIES_VISIBLE):
        sign   = 1 if (k % 2 == 0) else -1
        c0     = _hand_world_centroid(hand)
        d_side = (c0 - anchor).dot(side)
        new_loc = anchor \
                  + side * (sign * side_mm - d_side) \
                  + up   * lift_mm \
                  - cam_fwd * (FRONT_MM + 0.02 * length)
        hand.set_location(new_loc)
        bpy.context.view_layer.update()
        if _on_screen(hand, scene, cam):
            break

    _nudge_to_surface(hand, tool, up, side, anchor, side_mm, lift_mm)

    try:
        hand.blender_obj.parent = tool.blender_obj
        hand.blender_obj.matrix_parent_inverse = tool.blender_obj.matrix_world.inverted()
    except:
        pass
    hand.hide(False)

# =========================
# Keypoints
# =========================

def _tool_keypoints(tool_name, tool_obj, scene, cam, W, H):
    base = os.path.basename(tool_name).lower()
    kind = "NH" if "nh" in base else "T"

    shaft, up, side, base_w, tip_w, length, thick = _tool_axes_and_endpoints_world(tool_obj, kind)

    M = np.asarray(tool_obj.get_local2world_mat(), dtype=float)
    V = np.array([v.co[:] for v in tool_obj.get_mesh().vertices], dtype=float)
    Vw = (M @ np.c_[V, np.ones(len(V))].T).T[:, :3]
    cen_w = Vw.mean(0)

    side_vec = np.array([side.x, side.y, side.z], dtype=float)
    projs = (Vw - cen_w) @ side_vec
    left_w  = Vw[projs < 0] if np.any(projs < 0) else Vw
    right_w = Vw[projs >= 0] if np.any(projs >= 0) else Vw

    def nearest(pts, target):
        if len(pts) == 0:
            return target
        d2 = np.sum((pts - np.asarray(target))**2, axis=1)
        return pts[int(np.argmin(d2))]

    kps = {}
    kps["tip_left"]  = _project_point(nearest(left_w,  [tip_w.x,  tip_w.y,  tip_w.z]),  scene, cam, W, H)
    kps["tip_right"] = _project_point(nearest(right_w, [tip_w.x,  tip_w.y,  tip_w.z]), scene, cam, W, H)

    center_w = cen_w
    ortho = np.cross([shaft.x, shaft.y, shaft.z], [up.x, up.y, up.z])
    ortho = ortho / (np.linalg.norm(ortho) + 1e-9)
    stem_l = center_w - 0.015 * ortho
    stem_r = center_w + 0.015 * ortho
    kps["shaft_left"]  = _project_point(nearest(left_w,  stem_l),  scene, cam, W, H)
    kps["shaft_right"] = _project_point(nearest(right_w, stem_r), scene, cam, W, H)

    if kind == "NH":
        l, r = _ring_points_world(Vw.astype(np.float32), base_w, shaft, side_vec.astype(np.float32), length)
        kps["ring_left"]  = _project_point(l, scene, cam, W, H)
        kps["ring_right"] = _project_point(r, scene, cam, W, H)
    else:
        kps["base"] = _project_point([base_w.x, base_w.y, base_w.z], scene, cam, W, H)

    if all(v[2] == 0 for v in kps.values()):
        kps = {
            "tip":  _project_point([tip_w.x,  tip_w.y,  tip_w.z],  scene, cam, W, H),
            "base": _project_point([base_w.x, base_w.y, base_w.z], scene, cam, W, H),
        }
    return kps

# =========================
# Scene setup
# =========================

def load_objects(tools_base_folder, right_hand_path):
    nh_files = sorted(glob.glob(os.path.join(tools_base_folder, "needle_holder", "*.obj")))
    tw_files = sorted(glob.glob(os.path.join(tools_base_folder, "tweezers", "*.obj")))
    needle_holders, tweezers = [], []

    for f in nh_files:
        o = bproc.loader.load_obj(f)[0]
        o.set_cp("category_id", 1)
        o.set_cp("cp_physics", False)
        o.set_scale([1.5, 1.5, 1.5])
        o.set_name(os.path.basename(f))
        o.hide(); needle_holders.append(o)

    for f in tw_files:
        o = bproc.loader.load_obj(f)[0]
        o.set_cp("category_id", 2)
        o.set_cp("cp_physics", False)
        o.set_scale([1.5, 1.5, 1.5])
        o.set_name(os.path.basename(f))
        o.hide(); tweezers.append(o)

    parts = bproc.loader.load_obj(right_hand_path)
    right_hand = _largest_mesh(parts)
    for p in parts:
        if p != right_hand:
            p.hide()
    right_hand.set_cp("cp_physics", False)
    right_hand.set_name("RightHand")

    left_hand = right_hand.duplicate()
    _mirror_mesh_inplace_x(left_hand)
    left_hand.set_name("LeftHand")

    add_wrist_stub(right_hand)
    add_wrist_stub(left_hand)

    for o in needle_holders + tweezers:
        apply_surgical_metal(o)

    right_hand.hide(False)
    left_hand.hide(False)
    return needle_holders, tweezers, right_hand, left_hand

def load_camera_params(camera_params_path):
    with open(camera_params_path, "r") as f:
        p = json.load(f)
    K = np.array([[p["fx"], 0, p["cx"]],
                  [0, p["fy"], p["cy"]],
                  [0, 0, 1]])
    CameraUtility.set_intrinsics_from_K_matrix(K, p["width"], p["height"])
    camera_location = [0, 0, 20]
    R = bproc.camera.rotation_from_forward_vec([-x for x in camera_location], inplane_rot=0)
    cam2world = bproc.math.build_transformation_mat(camera_location, R)
    return cam2world, p["width"], p["height"]

def set_frame_positions(needle_holders, tweezers, right_hand, left_hand, scene, cam, W, H):
    nh = random.choice(needle_holders); nh.hide(False)
    nh_loc = np.random.uniform([-3, -1.2, -1], [-1, 0.2, 1])
    nh_rot = tuple(np.random.uniform([0, 0, (-5/8)*pi], [0, (1/2)*pi, (-1/4)*pi]))
    nh.set_rotation_euler(nh_rot); nh.set_location(nh_loc)

    tw = random.choice(tweezers); tw.hide(False)
    tw_loc = np.random.uniform([-1, -0.7, -1], [-0.5, 1.5, 1])
    tw_rot = tuple(np.random.uniform([0, 0, (10/12)*pi], [0, (1/2)*pi, (13/12)*pi]))
    tw.set_rotation_euler(tw_rot); tw.set_location(tw_loc)

    _align_and_scale_hand_to_tool(left_hand, nh, kind="NH", scene=scene, cam=cam)
    _align_and_scale_hand_to_tool(right_hand,  tw, kind="T",  scene=scene, cam=cam)

    ann = []
    ann.append({
        "name": nh.get_name(),
        "class_id": 1,
        "keypoints": _tool_keypoints(nh.get_name(), nh, scene, cam, W, H),
        "bbox": _bbox_norm_from_mesh(nh, scene, cam, W, H)
    })
    ann.append({
        "name": tw.get_name(),
        "class_id": 2,
        "keypoints": _tool_keypoints(tw.get_name(), tw, scene, cam, W, H),
        "bbox": _bbox_norm_from_mesh(tw, scene, cam, W, H)
    })
    return nh, tw, ann

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_dir',        default=TOOLS_DEFAULT_FOLDER)
    parser.add_argument('--camera_params',  default=CAMERA_PARAMS_DEFAULT)
    parser.add_argument('--output_dir',     default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--num_images',     type=int, default=NUM_IMAGES)
    parser.add_argument('--hdri_root',      default="/datashare/project/haven/hdris")
    parser.add_argument('--seed',           type=int, default=None, help="Random seed for repeatability")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔒 Deterministic run with seed={args.seed}")

    hdr_files = []
    for root, _, files in os.walk(args.hdri_root):
        for f in files:
            if f.lower().endswith((".hdr", ".exr")):
                hdr_files.append(os.path.join(root, f))
    if not hdr_files:
        raise RuntimeError(f"No HDRI files found under {args.hdri_root}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    keypoints_dir = os.path.join(args.output_dir, "keypoints")
    os.makedirs(keypoints_dir, exist_ok=True)

    bproc.init()
    nhs, tws, right_hand, left_hand = load_objects(args.obj_dir, RIGHT_HAND_FILE)
    cam2world, W, H = load_camera_params(args.camera_params)

    rig = setup_natural_hdr_lighting(hdr_files)
    apply_glove_shader(right_hand, tint=rig["tint"])
    apply_glove_shader(left_hand,  tint=rig["tint"])

    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance", "name"],
        default_values={"category_id": 0}
    )

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    cam = scene.camera

    c = scene.cycles
    c.device = "CPU"
    c.samples = 24
    c.preview_samples = 8
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.max_bounces = 2
    c.diffuse_bounces = 1
    c.glossy_bounces = 2
    c.transparent_max_bounces = 2
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.light_sampling_threshold = 0.01
    c.blur_glossy = 0.25

    bproc.renderer.set_max_amount_of_samples(c.samples)

    scene.render.use_stamp = False
    scene.render.use_stamp_note = False
    scene.render.stamp_font_size = 12
    scene.render.stamp_note_text = f"RUN:{os.path.basename(args.output_dir)} S:{c.samples}"

    scene.render.use_persistent_data = True
    if hasattr(scene.render, "tile_x"):
        scene.render.tile_x = scene.render.tile_y = 256
    if hasattr(scene.cycles, "tile_size"):
        scene.cycles.tile_size = 256

    save_root = os.path.join(args.output_dir, "coco_data")

    for i in range(args.num_images):
        bproc.utility.reset_keyframes()
        cam.matrix_world = Matrix(cam2world)
        print(f"RUN:{os.path.basename(args.output_dir)} S:{scene.cycles.samples} frame {i}")

        # Small, bounded energy jitter (seed-controlled if provided)
        rig["key"].data.energy *= random.uniform(0.95, 1.05)
        rig["fill"].data.energy = rig["key"].data.energy * 0.08
        rig["rim"].data.energy  = rig["key"].data.energy * 0.06

        nh, tw, frame_ann = set_frame_positions(nhs, tws, right_hand, left_hand, scene, cam, W, H)
        bproc.camera.add_camera_pose(cam2world)

        try:
            r = rig["map"].inputs["Rotation"].default_value
            r[2] += random.uniform(-0.25, 0.25)
            rig["map"].inputs["Rotation"].default_value = r
        except Exception:
            pass

        data = bproc.renderer.render()
        bproc.writer.write_coco_annotations(
            save_root,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            mask_encoding_format="rle",
            append_to_existing_output=True
        )

        with open(os.path.join(keypoints_dir, f"{i:06}.json"), "w") as f:
            json.dump(frame_ann, f, indent=2)

        nh.hide(); tw.hide()
        try: right_hand.blender_obj.parent = None
        except: pass
        try: left_hand.blender_obj.parent = None
        except: pass

    print(f"✅ Images: {os.path.join(save_root, 'images')}")
    print(f"✅ COCO:   {os.path.join(save_root, 'coco_annotations.json')}")
    print(f"✅ KPs:    {keypoints_dir}")

if __name__ == "__main__":
    main()