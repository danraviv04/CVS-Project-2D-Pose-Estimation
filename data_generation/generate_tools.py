import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
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

def get_left_right_split_labels(centered_vertices: np.ndarray):
    # Use PCA to find principal directions
    cov = np.cov(centered_vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]  # columns: [PC0, PC1, PC2]

    # PC1 is usually the left/right axis (symmetry axis)
    left_right_axis = eigvecs[:, 1]
    projections = centered_vertices @ left_right_axis
    labels = (projections > 0).astype(int)

    return labels, left_right_axis, eigvecs

# def plot_split_debug(title, local_vertices, labels, eigvecs, output_dir, tool_name):
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(local_vertices[:, 0], local_vertices[:, 1], local_vertices[:, 2],
#                c=labels, cmap='coolwarm', s=1)
#     origin = np.zeros(3)
#     for i, vec in enumerate(eigvecs.T):
#         ax.quiver(*origin, *vec, color=['r', 'g', 'b'][i], linewidth=2, label=f"PC{i}")
#     ax.set_title(title)
#     ax.legend()
#     ax.view_init(elev=30, azim=30)
#     os.makedirs(os.path.join(output_dir, "debug_split"), exist_ok=True)
#     filename = os.path.join(output_dir, "debug_split", f"{tool_name}_{title.replace(' ', '_')}.png")
#     plt.savefig(filename, dpi=150)
#     plt.close()

def get_tool_keypoints_local(tool_name, obj):
    base = os.path.basename(tool_name).lower()
    mesh = obj.get_mesh()
    vertices = np.array([v.co[:] for v in mesh.vertices])
    mean = np.mean(vertices, axis=0)
    centered = vertices - mean

    # Left-right split
    labels, split_axis, eigvecs = get_left_right_split_labels(centered)
    local_vertices = centered @ eigvecs
    #plot_split_debug("Left-Right Split", local_vertices, labels, eigvecs, "", os.path.basename(tool_name))

    part_0 = vertices[labels == 0]
    part_1 = vertices[labels == 1]

    if np.mean((part_0 - mean) @ split_axis) > np.mean((part_1 - mean) @ split_axis):
        right, left = part_0, part_1
    else:
        right, left = part_1, part_0

    local_right = (right - mean) @ eigvecs
    local_left = (left - mean) @ eigvecs

    major = eigvecs[:, 0]  # PC0 = longest axis

    if "nh" in base:
        print(f"📌 Needle Holder layout (left-right split): {tool_name}")

        shaft_right = right[np.argmin(np.abs((right - mean) @ major))]
        shaft_left = left[np.argmin(np.abs((left - mean) @ major))]

        def find_tip(jaw, shaft):
            projections = (jaw - mean) @ major
            for pt in jaw[np.argsort(projections)[::-1]]:
                if np.linalg.norm(pt - shaft) > 0.005:
                    return pt
            return jaw[np.argmax(projections)]

        raw_tip_right = find_tip(right, shaft_right)
        raw_tip_left = find_tip(left, shaft_left)


        raw_ring_right = right[np.argmax(np.linalg.norm(right - mean, axis=1))]
        raw_ring_left = left[np.argmax(np.linalg.norm(left - mean, axis=1))]

        # def disambiguate_tip_ring(tip_cand, ring_cand):
        #     return (tip_cand, ring_cand) if np.linalg.norm(tip_cand - mean) < np.linalg.norm(ring_cand - mean) else (ring_cand, tip_cand)
        # 
        # ring_right, tip_right = disambiguate_tip_ring(tip_right, ring_right)
        # ring_left, tip_left = disambiguate_tip_ring(tip_left, ring_left)
        
        def disambiguate_tip_ring(p1, p2, prong_vertices, prong_axis):
            """
            Decide which point is tip and which is ring, based on shape cues.
            Arguments:
                p1, p2: candidate points (3D coords)
                prong_vertices: vertices of the prong (Nx3 array)
                prong_axis: major axis of the prong (usually PC0 or PC1)
            """
            def compute_pointiness_score(p):
                # Vector from mean to p
                v = p - prong_vertices.mean(axis=0)
                v_norm = v / (np.linalg.norm(v) + 1e-8)

                # Alignment with axis (dot product close to 1 → pointy along prong)
                alignment = np.abs(np.dot(v_norm, prong_axis))

                # Distance from centroid (favor farther points)
                dist = np.linalg.norm(v)

                return alignment + 0.2 * dist  # Weight can be tuned

            s1 = compute_pointiness_score(p1)
            s2 = compute_pointiness_score(p2)

            if s1 > s2:
                return p1, p2  # p1 = tip, p2 = ring
            else:
                return p2, p1  # p2 = tip, p1 = ring

                
        tip_right, ring_right = disambiguate_tip_ring(raw_tip_right, raw_ring_right, right, major)
        tip_left, ring_left = disambiguate_tip_ring(raw_tip_left, raw_ring_left, left, major)


        return {
            "tip_right": list(tip_right),
            "tip_left": list(tip_left),
            "shaft_right": list(shaft_right),
            "shaft_left": list(shaft_left),
            "ring_right": list(ring_right),
            "ring_left": list(ring_left),
        }

    elif base.startswith("t") and not base.startswith("nh"):
        print(f"📌 Tweezer layout (left-right split): {tool_name}")

        tip_right = right[np.argmax(local_right[:, 0])]
        tip_left = left[np.argmax(local_left[:, 0])]

        def get_midpoint_along_prong(local_prong, prong_vertices):
            fwd = local_prong[:, 0]
            target = fwd.min() + 0.50 * (fwd.max() - fwd.min())
            return prong_vertices[np.argmin(np.abs(fwd - target))]

        stem_right = get_midpoint_along_prong(local_right, right)
        stem_left = get_midpoint_along_prong(local_left, left)

        extent_major = centered @ major
        base_pt = mean - major * 0.5 * (extent_major.max() - extent_major.min())

        return {
            "tip_right": list(tip_right),
            "tip_left": list(tip_left),
            "stem_right": list(stem_right),
            "stem_left": list(stem_left),
            "base": list(base_pt),
        }

    else:
        print(f"⚠️ Fallback to corners for: {tool_name}")
        return {f"corner_{i}": coord for i, coord in enumerate(obj.get_bound_box())}

def object_fully_in_frame(obj, cam2world, cam_obj, scene, margin=0.05):
    mat = obj.get_local2world_mat()
    for corner in obj.get_bound_box():
        world_corner = mat @ Vector(list(corner) + [1])
        if not isinstance(world_corner, Vector):
            world_corner = Vector(world_corner[:3])
        proj = world_to_camera_view(scene, cam_obj, world_corner)
        if not (margin <= proj.x <= 1 - margin and margin <= proj.y <= 1 - margin and proj.z >= 0):
            return False
    return True

# def project_3d_to_2d(world_point, scene, cam_obj):
#     if isinstance(world_point, np.ndarray):
#         world_point = Vector(world_point.tolist())
#     co_2d = world_to_camera_view(scene, cam_obj, world_point)
#     if not (0.0 <= co_2d.x <= 1.0 and 0.0 <= co_2d.y <= 1.0 and co_2d.z >= 0.0):
#         return None
#     render = scene.render
#     return [int(co_2d.x * render.resolution_x), int((1 - co_2d.y) * render.resolution_y)]
def project_3d_to_2d(world_point, scene, cam_obj):
    if isinstance(world_point, np.ndarray):
        world_point = Vector(world_point.tolist())
    co_2d = world_to_camera_view(scene, cam_obj, world_point)
    margin = 0.05
    if not (-margin <= co_2d.x <= 1.0 + margin and -margin <= co_2d.y <= 1.0 + margin and co_2d.z >= -0.05):
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

parser = argparse.ArgumentParser()
parser.add_argument('--obj_dir', default="/datashare/project/surgical_tools_models")
parser.add_argument('--camera_params', default="/datashare/project/camera.json")
parser.add_argument('--output_dir', default="/home/student/project/output")
parser.add_argument('--num_images', type=int, default=5)
args = parser.parse_args()

bproc.init()

if bpy.context.scene.camera is None:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

with open(args.camera_params, "r") as file:
    camera_params = json.load(file)
K = np.array([[camera_params["fx"], 0, camera_params["cx"]],
              [0, camera_params["fy"], camera_params["cy"]],
              [0, 0, 1]])
CameraUtility.set_intrinsics_from_K_matrix(K, camera_params["width"], camera_params["height"])

obj_files = glob.glob(os.path.join(args.obj_dir, "*/*.obj"))
assert obj_files, "No .obj models found!"

bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.set_max_amount_of_samples(128)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

for img_idx in range(args.num_images):
    bproc.utility.reset_keyframes()
    for obj in bpy.data.objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    num_tools = random.choice([1, 2])
    chosen_objs = random.sample(obj_files, k=num_tools)
    tool_locations = []
    loaded_objs = []

    for obj_path in chosen_objs:
        tool_name = os.path.basename(obj_path)
        objs = bproc.loader.load_obj(obj_path)
        obj = objs[0]
        obj.set_cp("category_id", 1)
        obj.set_name(tool_name)

        mat = obj.get_materials()[0] if obj.get_materials() else None
        if mat:
            mat.set_principled_shader_value("Roughness", 0.2)
            mat.set_principled_shader_value("Metallic", 1)

        location = maybe_overlapping_location(tool_locations)
        tool_locations.append(location)
        obj.set_location(location)
        obj.set_rotation_euler(Euler([random.uniform(0, 2*np.pi) for _ in range(3)], 'XYZ'))

        bbox = np.array(obj.get_bound_box())
        bbox_min = bbox.min(axis=0)
        bbox_max = bbox.max(axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        scale_factor = 0.22 / diag
        obj.set_scale([scale_factor] * 3)

        loaded_objs.append((obj, tool_name))

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(bproc.sampler.shell([0, 0, 0], 1.2, 2.0, 30, 85))
    light.set_energy(random.uniform(800, 1500))

    scene = bpy.context.scene
    cam_obj = scene.camera

    for _ in range(200):
        cam_loc = bproc.sampler.shell([0, 0, 0], 0.9, 1.3, -30, 60)
        look_at = np.array([0, 0, 0])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_loc)
        cam2world = bproc.math.build_transformation_mat(cam_loc, rotation_matrix)

        if sum([obj in bproc.camera.visible_objects(cam2world) for obj, _ in loaded_objs]) >= len(loaded_objs) - 1:
            bproc.camera.add_camera_pose(cam2world)
            break
    else:
        print(f"❌ No valid pose for image {img_idx}, skipping.")
        continue

    data = bproc.renderer.render()

    keypoints_anno = []
    for obj, tool_name in loaded_objs:
        local_keypoints = get_tool_keypoints_local(tool_name, obj)
        named_kps = {}
        for name, kp_local in local_keypoints.items():
            kp_world = obj.get_local2world_mat() @ Vector(list(kp_local) + [1])
            proj = project_3d_to_2d(kp_world, scene, cam_obj)
            if proj:
                named_kps[name] = [proj[0], proj[1], 2]
            else:
                named_kps[name] = [0, 0, 0]

        visible_count = sum(1 for v in named_kps.values() if v[2] > 0)
        print(f"🏋️ {obj.get_name()} — {visible_count}/{len(named_kps)} keypoints visible")
        keypoints_anno.append({
            "name": obj.get_name(),
            "keypoints": named_kps
        })
        
        # Diagnostic: print vertical range in pixel space
        ys = [v[1] for v in named_kps.values() if v[2] > 0]
        if ys:
            y_range = max(ys) - min(ys)
            print(f"📏 {tool_name} Y-range in 2D: {y_range:.2f} pixels")
        else:
            print(f"⚠️ No visible keypoints for {tool_name} to measure Y-range")



    keypoints_dir = os.path.join(args.output_dir, 'keypoints')
    os.makedirs(keypoints_dir, exist_ok=True)
    with open(os.path.join(keypoints_dir, f"{img_idx:06}.json"), "w") as f:
        json.dump(keypoints_anno, f, indent=2)

    bproc.writer.write_coco_annotations(
        os.path.join(args.output_dir, 'coco_data'),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        mask_encoding_format="rle",
        append_to_existing_output=True
    )

    print(f"✅ Rendered image {img_idx + 1}/{args.num_images}")