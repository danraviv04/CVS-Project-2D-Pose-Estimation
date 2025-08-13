import blenderproc as bproc
import bpy
import numpy as np
import argparse
import random
import os
import json
import glob
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# ========== HDR LIGHTING ========== #
import bpy
import random
import os
from math import radians

def setup_or_lighting(hdri_files):
    """
    Sets up a random HDRI from hdri_path (recursive search) + a surgical-style spotlight.
    """
    # --- Find all HDR/EXR files in subfolders ---

    chosen_hdri = random.choice(hdr_files)
    print(f"🌅 Using HDRI: {chosen_hdri}")

    # --- Clear existing world nodes ---
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    # --- Create nodes ---
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    env_tex = nt.nodes.new("ShaderNodeTexEnvironment")
    bg = nt.nodes.new("ShaderNodeBackground")
    world_out = nt.nodes.new("ShaderNodeOutputWorld")

    # Load HDRI
    env_tex.image = bpy.data.images.load(chosen_hdri)

    # Random rotation of HDRI
    mapping.inputs['Rotation'].default_value[2] = random.uniform(0, 6.2831)

    # Link nodes
    nt.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    nt.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    nt.links.new(env_tex.outputs['Color'], bg.inputs['Color'])
    nt.links.new(bg.outputs['Background'], world_out.inputs['Surface'])

    # --- Add a surgical-style spotlight ---
    bpy.ops.object.light_add(type='SPOT', location=(0, 0, 1.2))
    spot = bpy.context.object
    spot.data.energy = 1000
    spot.data.spot_size = radians(50)
    spot.data.spot_blend = 0.3
    spot.data.use_shadow = True
    spot.data.shadow_soft_size = 0.05

    # Point the spot downwards
    spot.rotation_euler = (radians(90), 0, 0)

    print("✅ HDRI and OR-style lighting setup complete.")

# ========== LEFT/RIGHT SPLIT ========== #
def get_left_right_split_labels(centered_vertices: np.ndarray, jitter=0.001):
    """Split vertices into left/right using PCA + random flip."""
    # Add small random perturbation for stability (optional)
    perturbed = centered_vertices + np.random.normal(scale=jitter, size=centered_vertices.shape)

    cov = np.cov(perturbed.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    split_axis = eigvecs[:, 1]  # Use PC1 for symmetry axis

    # # Randomly flip axis to vary left/right assignment
    # if random.random() < 0.5:
    #     split_axis = -split_axis

    projections = centered_vertices @ split_axis
    labels = (projections > 0).astype(int)
    return labels, split_axis, eigvecs


# ========== DEBUG PLOT ========== #
def plot_split_debug(title, local_vertices, labels, eigvecs, output_dir, tool_name, frame_idx=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(local_vertices[:, 0], local_vertices[:, 1], local_vertices[:, 2],
               c=labels, cmap='coolwarm', s=2)
    origin = np.zeros(3)
    for i, vec in enumerate(eigvecs.T):
        ax.quiver(*origin, *vec, color=['r', 'g', 'b'][i], linewidth=2, label=f"PC{i}")
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=30, azim=30)

    os.makedirs(os.path.join(output_dir, "debug_split"), exist_ok=True)
    frame_str = f"_frame{frame_idx:06d}" if frame_idx is not None else ""
    filename = os.path.join(output_dir, "debug_split",
                            f"{os.path.splitext(tool_name)[0]}{frame_str}_{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.close()


# ========== TOOL KEYPOINTS ========== #

def get_tool_keypoints_local(tool_name, obj, frame_idx):
    base = os.path.basename(tool_name).lower()
    mesh = obj.get_mesh()
    vertices = np.array([v.co[:] for v in mesh.vertices])
    mean = np.mean(vertices, axis=0)
    centered = vertices - mean
    labels, split_axis, eigvecs = get_left_right_split_labels(centered)
    local_vertices = centered @ eigvecs

    # Split
    projections = centered @ split_axis
    left = vertices[projections < 0]
    right = vertices[projections >= 0]
    local_left = (left - mean) @ eigvecs
    local_right = (right - mean) @ eigvecs
    major = eigvecs[:, 0]

    if "nh" in base:
        # Get shaft: closest to mean along major axis
        shaft_right = right[np.argmin(np.abs((right - mean) @ major))]
        shaft_left = left[np.argmin(np.abs((left - mean) @ major))]

        # TIP = extremal in -major axis direction
        tip_right = right[np.argmin((right - mean) @ major)]
        tip_left = left[np.argmin((left - mean) @ major)]

        # RING = extremal in +major axis direction
        ring_right = right[np.argmax((right - mean) @ major)]
        ring_left = left[np.argmax((left - mean) @ major)]

        # === Stability check ===
        tip_score = (tip_right - mean) @ major
        ring_score = (ring_right - mean) @ major
        score_diff = abs(tip_score - ring_score)

        if score_diff < 0.01:
            print(f"⚠️ [Frame {frame_idx}] NH tip/ring too close: Δ = {score_diff:.5f} — tool: {tool_name}")

        return {
            "tip_right": list(tip_right),
            "tip_left": list(tip_left),
            "shaft_right": list(shaft_right),
            "shaft_left": list(shaft_left),
            "ring_right": list(ring_right),
            "ring_left": list(ring_left),
        }

    elif base.startswith("t") and not base.startswith("nh"):
        
        def get_midpoint_along_prong(local_prong, prong_vertices):
            fwd = local_prong[:, 0]
            target = fwd.min() + 0.5 * (fwd.max() - fwd.min())
            return prong_vertices[np.argmin(np.abs(fwd - target))]
        
        tip_right = right[np.argmax(local_right[:, 0])]
        tip_left = left[np.argmax(local_left[:, 0])]

        # Push shaft points apart along orthogonal symmetry axis
        orthogonal_axis = eigvecs[:, 1]
        offset = 0.015  # Tune this for more/less separation
        stem_right = mean + orthogonal_axis * offset
        stem_left = mean - orthogonal_axis * offset

        extent_major = centered @ major
        base_pt = mean - major * 0.5 * (extent_major.max() - extent_major.min())

        return {
            "tip_right": list(tip_right),
            "tip_left": list(tip_left),
            "shaft_right": list(stem_right),
            "shaft_left": list(stem_left),
            "base": list(base_pt),
        }


    return {f"corner_{i}": coord for i, coord in enumerate(obj.get_bound_box())}

def project_3d_to_2d(world_point, scene, cam_obj):
    if isinstance(world_point, np.ndarray):
        world_point = Vector(world_point.tolist())
    co_2d = world_to_camera_view(scene, cam_obj, world_point)
    # margin = 0.05
    # if not (-margin <= co_2d.x <= 1.0 + margin and -margin <= co_2d.y <= 1.0 + margin and co_2d.z >= -0.05):
    #     return None
    margin = 0.02  # tighten up
    if not (margin <= co_2d.x <= 1.0 - margin and margin <= co_2d.y <= 1.0 - margin and co_2d.z >= 0):
        return None
    render = scene.render
    x = int(np.clip(co_2d.x, 0, 1) * render.resolution_x)
    y = int((1 - np.clip(co_2d.y, 0, 1)) * render.resolution_y)
    return [x, y]

def maybe_overlapping_location(existing_locations, overlap_prob=0.05, min_distance=0.3, max_attempts=100):
    if random.random() < overlap_prob and existing_locations:
        base = random.choice(existing_locations)
        return base + np.array([random.uniform(-0.05, 0.05), 0, random.uniform(-0.05, 0.05)])
    for _ in range(max_attempts):
        loc = np.array([random.uniform(-0.1, 0.1), 0, random.uniform(-0.1, 0.1)])
        if all(np.linalg.norm(loc - prev) >= min_distance for prev in existing_locations):
            return loc
    return loc

def is_mostly_in_frame(obj, cam_obj, scene, required_visible=5):
    mat = obj.get_local2world_mat()
    corners = obj.get_bound_box()
    visible = 0
    for corner in corners:
        corner_vec = Vector(corner).to_4d()         # ensure 4D homogeneous coordinate
        world_corner = mat @ corner_vec             # apply transformation
        proj = world_to_camera_view(scene, cam_obj, Vector(world_corner[:3]))  # convert to 3D Vector
        if 0 <= proj.x <= 1 and 0 <= proj.y <= 1 and proj.z >= 0:
            visible += 1
    return visible >= required_visible

def apply_random_pose_jitter(obj, location_range=0.15, z_rot_range=2*np.pi):
    """
    Applies small random translation and full rotation to the object.
    """
    # Random small XY translation (to simulate tool position changes)
    jitter_x = random.uniform(-location_range, location_range)
    jitter_y = random.uniform(-location_range, location_range)
    jitter_z = 0.0  # Don't move in Z; keep object on the same plane

    current_loc = obj.get_location()
    obj.set_location(Vector((
        current_loc[0] + jitter_x,
        current_loc[1] + jitter_y,
        current_loc[2] + jitter_z
    )))

    # Random rotation (pitch, yaw, roll)
    rot_x = random.uniform(0, np.pi * 2)
    rot_y = random.uniform(0, np.pi * 2)
    rot_z = random.uniform(0, z_rot_range)

    obj.set_rotation_euler(Euler((rot_x, rot_y, rot_z), 'XYZ'))
    
def smart_scale_tool(obj, camera_distance=1.0, target_screen_fraction=0.23, safety_margin=0.9):
    bbox = np.array(obj.get_bound_box())
    extent = bbox.max(axis=0) - bbox.min(axis=0)
    size = np.linalg.norm(extent)

    # Approximate scale to hit the same visual footprint for typical FoV
    base_scale = target_screen_fraction / size

    # Prevent large tools from going out of frame (limit X or Y extent)
    max_dim = max(extent[0], extent[2])  # horizontal space used on screen
    if max_dim * base_scale > safety_margin:
        base_scale = safety_margin / max_dim

    obj.set_scale([base_scale] * 3)

from bpy_extras.object_utils import world_to_camera_view

from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# def get_projected_bbox(obj, scene, cam_obj, image_width, image_height, tool_name=None):

#     mesh = obj.get_mesh()
#     mat = obj.get_local2world_mat()

#     coords_2d = []

#     for v in mesh.vertices:
#         world_coord = mat @ v.co.to_4d()
#         proj = world_to_camera_view(scene, cam_obj, Vector(world_coord[:3]))

#         x = proj.x * image_width
#         y = (1.0 - proj.y) * image_height  # Flip Y

#         if 0 <= x <= image_width and 0 <= y <= image_height:
#             coords_2d.append((x, y))

#     if len(coords_2d) < 3:
#         return [0.0, 0.0, 0.0, 0.0]  # Not enough visible points

#     coords = np.array(coords_2d)
#     x_min, y_min = coords.min(axis=0)
#     x_max, y_max = coords.max(axis=0)

#     # --- Padding logic ---
#     base_pad_ratio = 0.05
#     extra_pad = 0.03 if tool_name and "t" in os.path.basename(tool_name).lower() and "nh" not in tool_name.lower() else 0.0
#     pad_x = (base_pad_ratio + extra_pad) * (x_max - x_min)
#     pad_y = (base_pad_ratio + extra_pad) * (y_max - y_min)

#     x_min = max(0.0, x_min - pad_x)
#     y_min = max(0.0, y_min - pad_y)
#     x_max = min(image_width, x_max + pad_x)
#     y_max = min(image_height, y_max + pad_y)

#     # --- Normalize for YOLO ---
#     cx = (x_min + x_max) / 2 / image_width
#     cy = (y_min + y_max) / 2 / image_height
#     w = (x_max - x_min) / image_width
#     h = (y_max - y_min) / image_height

#     # Optional: clamp min size
#     min_size = 1.0 / image_width  # ~1 pixel
#     w = max(w, min_size)
#     h = max(h, min_size)

#     return [cx, cy, w, h]

def get_projected_bbox(obj, scene, cam_obj, image_width, image_height, tool_name=None, pad_ratio=0.05):
    """
    Returns YOLO-normalized (cx, cy, w, h) from the projected mesh verts.
    Only uses verts visible to the camera (0<=x,y<=1 and z>=0).
    """
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    mesh = obj.get_mesh()
    mat  = obj.get_local2world_mat()

    coords_px = []
    for v in mesh.vertices:
        w = mat @ v.co.to_4d()
        cv = world_to_camera_view(scene, cam_obj, Vector(w[:3]))
        if 0.0 <= cv.x <= 1.0 and 0.0 <= cv.y <= 1.0 and cv.z >= 0.0:
            x = cv.x * image_width
            y = (1.0 - cv.y) * image_height  # flip Y to pixels
            coords_px.append((x, y))

    # Not enough visible points → no box
    if len(coords_px) < 3:
        return [0.0, 0.0, 0.0, 0.0]

    xs, ys = zip(*coords_px)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    # Padding
    px = pad_ratio * (x1 - x0)
    py = pad_ratio * (y1 - y0)
    x0 -= px; x1 += px
    y0 -= py; y1 += py

    # Clip to frame
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(float(image_width),  x1)
    y1 = min(float(image_height), y1)

    # Degenerate guard (~1px minimum)
    if x1 <= x0: x1 = min(x0 + 1.0, image_width)
    if y1 <= y0: y1 = min(y0 + 1.0, image_height)

    # Normalize to YOLO (cx, cy, w, h)
    cx = ((x0 + x1) * 0.5) / image_width
    cy = ((y0 + y1) * 0.5) / image_height
    w  = (x1 - x0) / image_width
    h  = (y1 - y0) / image_height

    # Clamp tiny boxes to ~1px normalized
    w = max(w, 1.0 / image_width)
    h = max(h, 1.0 / image_height)
    return [cx, cy, w, h]

# === ARGUMENTS === #
parser = argparse.ArgumentParser()
parser.add_argument('--obj_dir', default="/datashare/project/surgical_tools_models")
parser.add_argument('--camera_params', default="/datashare/project/camera.json")
parser.add_argument('--output_dir', default="/home/student/project/output")
parser.add_argument('--hdri_root', default="/datashare/project/haven/hdris")
parser.add_argument('--num_images', type=int, default=5)
args = parser.parse_args()

hdr_files = []
for root, _, files in os.walk(args.hdri_root):
    for f in files:
        if f.lower().endswith((".hdr", ".exr")):
            hdr_files.append(os.path.join(root, f))
if not hdr_files:
    raise RuntimeError(f"No HDRI files found under {args.hdri_root}")

# === INIT === #
bproc.init()
# === CAMERA SETUP === #
if bpy.context.scene.camera is None:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

with open(args.camera_params, "r") as f:
    camera_params = json.load(f)
K = np.array([[camera_params["fx"], 0, camera_params["cx"]],
              [0, camera_params["fy"], camera_params["cy"]],
              [0, 0, 1]])
bproc.camera.set_intrinsics_from_K_matrix(K, camera_params["width"], camera_params["height"])

# === RENDER SETTINGS === #
bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.set_max_amount_of_samples(128)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# === TOOL LOADING === #
obj_files = glob.glob(os.path.join(args.obj_dir, "*/*.obj"))
assert obj_files, "No .obj models found!"

# === MAIN LOOP === #
for img_idx in range(args.num_images):
    bproc.utility.reset_keyframes()
    for obj in bpy.data.objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    cam_obj = scene.camera
    
    num_tools = random.choice([1, 2])
    chosen_objs = random.sample(obj_files, k=num_tools)
    tool_locations = []
    loaded_objs = []

    for obj_path in chosen_objs:
        tool_name = os.path.basename(obj_path)
        obj = bproc.loader.load_obj(obj_path)[0]

        class_id = 0 if "nh" in tool_name.lower() else 1
        obj.set_cp("category_id", class_id)

        mat = obj.get_materials()[0] if obj.get_materials() else None
        if mat:
            mat.set_principled_shader_value("Roughness", random.uniform(0.1, 0.4))
            mat.set_principled_shader_value("Metallic", 1.0)

        # location = bproc.sampler.upper_half_sphere(radius_min=0.1, radius_max=0.2)
        location = maybe_overlapping_location(tool_locations)
        obj.set_location(location)
        #obj.set_rotation_euler(Euler([random.uniform(0, 2*np.pi) for _ in range(3)], 'XYZ'))
        
        obj.set_rotation_euler(Euler([
            random.uniform(-0.3, 0.3),  # X (pitch)
            random.uniform(-0.3, 0.3),  # Y (roll)
            random.uniform(0, 2*np.pi)  # Z (spin)
        ], 'XYZ'))
        
        #apply_random_pose_jitter(obj)
        

        # bbox = np.array(obj.get_bound_box())
        # diag = np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0))
        # scale = 0.25 / diag
        # scale = min(scale, 1.0)  # Don't blow things up
        # scale = max(scale, 0.05)  # Don't shrink too small
        # scale *= random.uniform(0.95, 1.05)
        # obj.set_scale([scale]*3)
        
        bbox = np.array(obj.get_bound_box())
        diag = np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0))

        target_frac = np.random.choice([  # ~70% small, 25% medium, 5% big
            np.random.uniform(0.06, 0.12),  # small
            np.random.uniform(0.12, 0.20),  # medium
            np.random.uniform(0.20, 0.28)   # big (rare)
        ], p=[0.7, 0.25, 0.05])

        scale = max(0.03, min(1.0, target_frac / (diag if diag > 1e-6 else 1.0)))
        scale *= random.uniform(0.95, 1.05)
        obj.set_scale([scale]*3)
        
        loaded_objs.append((obj, tool_name))

    # Lighting
    setup_or_lighting(hdr_files)  # e.g. "/datashare/project/haven/hdris"
    # light = bproc.types.Light()
    # light.set_type("POINT")
    # light.set_location(bproc.sampler.shell([0, 0, 0], 1.2, 2.0, 30, 85))
    # light.set_energy(random.uniform(800, 1500))

    # Camera pose
    for _ in range(200):
        #cam_loc = bproc.sampler.shell([0, 0, 0], 0.9, 1.3, -30, 60)
        cam_loc = bproc.sampler.shell([0, 0, 0], 0.8, 1.1, -30, 60)
        look_at = np.array([0, 0, 0])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_loc)
        cam2world = bproc.math.build_transformation_mat(cam_loc, rotation_matrix)

        if sum([obj in bproc.camera.visible_objects(cam2world) for obj, _ in loaded_objs]) >= len(loaded_objs) - 1:
            bproc.camera.add_camera_pose(cam2world)
            break
    else:
        print(f"❌ No valid pose for image {img_idx}, skipping.")
        continue

    # Rendering
    data = bproc.renderer.render()

    # Save keypoints
    # keypoints_anno = []
    # for obj, tool_name in loaded_objs:
    #     local_kps = get_tool_keypoints_local(tool_name, obj, img_idx)
    #     named_kps = {}
    #     for name, kp_local in local_kps.items():
    #         kp_world = obj.get_local2world_mat() @ Vector(list(kp_local) + [1])
    #         proj = project_3d_to_2d(kp_world, scene, cam_obj)
    #         named_kps[name] = [proj[0], proj[1], 2] if proj else [0, 0, 0]

    #     keypoints_anno.append({
    #         "name": tool_name,
    #         "class_id": class_id,  # Save the class!
    #         "keypoints": named_kps
    #     })
    
    keypoints_anno = []
    for obj, tool_name in loaded_objs:
        local_kps = get_tool_keypoints_local(tool_name, obj, img_idx)
        named_kps = {}
        for name, kp_local in local_kps.items():
            kp_world = obj.get_local2world_mat() @ Vector(list(kp_local) + [1])
            proj = project_3d_to_2d(kp_world, scene, cam_obj)
            named_kps[name] = [proj[0], proj[1], 2] if proj else [0, 0, 0]
        
        W, H = camera_params["width"], camera_params["height"]
        bbox = get_projected_bbox(obj, scene, cam_obj, image_width=W, image_height=H, tool_name=tool_name)

        keypoints_anno.append({
            "name": tool_name,
            "class_id": class_id,
            "keypoints": named_kps,
            "bbox": bbox  # now from mesh
        })

    os.makedirs(os.path.join(args.output_dir, 'keypoints'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'keypoints', f"{img_idx:06}.json"), "w") as f:
        json.dump(keypoints_anno, f, indent=2)

    # COCO Annotations
    bproc.writer.write_coco_annotations(
        os.path.join(args.output_dir, 'coco_data'),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        mask_encoding_format="rle",
        append_to_existing_output=True
    )

    print(f"✅ Rendered image {img_idx + 1}/{args.num_images}")