import blenderproc as bproc
import numpy as np
import bpy
import os
import json
import random
import imageio.v2 as imageio
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view
from blenderproc.python.camera.CameraUtility import set_intrinsics_from_K_matrix
import argparse

def load_hdr_background(hdri_path):
    world = bpy.context.scene.world
    world.use_nodes = True
    world.cycles_visibility.camera = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    try:
        env_tex.image = bpy.data.images.load(hdri_path, check_existing=True)
        env_tex.image.colorspace_settings.name = 'sRGB'
    except Exception as e:
        print(f"Failed to load HDRI: {hdri_path}, error: {e}")
        return

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = random.uniform(1.0, 2.5)

    output = nodes.new(type="ShaderNodeOutputWorld")

    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

def enforce_principled_shader(obj):
    for mat in obj.get_materials():
        b_mat = mat.blender_obj

        if b_mat.use_nodes and b_mat.node_tree and len(b_mat.node_tree.nodes) > 0:
            continue

        b_mat.use_nodes = True
        nodes = b_mat.node_tree.nodes
        links = b_mat.node_tree.links

        nodes.clear()

        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.2
        bsdf.inputs["Specular"].default_value = 0.5

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

def is_fully_in_frame(obj, cam2world):
    return True
    scene = bpy.context.scene
    camera = scene.camera
    for v in obj.blender_obj.data.vertices:
        co_ndc = world_to_camera_view(scene, camera, obj.blender_obj.matrix_world @ v.co)
        if not (0.0 <= co_ndc.x <= 1.0 and 0.0 <= co_ndc.y <= 1.0 and 0.0 <= co_ndc.z <= 1.0):
            return False
    return True

def get_bounding_radius(obj):
    bbox = [obj.blender_obj.matrix_world @ Vector(corner) for corner in obj.blender_obj.bound_box]
    center = sum(bbox, Vector()) / 8
    return max((v - center).length for v in bbox)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools_dir", default="/datashare/project/surgical_tools_models")
    parser.add_argument("--hdris_dir", default="/datashare/project/haven/hdris")
    parser.add_argument("--camera_json", default="/datashare/project/camera.json")
    parser.add_argument("--output_dir", default="/home/student/project/output")
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)

    bproc.init()

    tools = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.tools_dir)
        for f in files if f.endswith(".obj")
    ]
    if not tools:
        raise RuntimeError("No .obj files found!")

    hdrs = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.hdris_dir)
        for f in files if f.endswith(".hdr")
    ]
    if not hdrs:
        raise RuntimeError("No HDRI files found!")

    with open(args.camera_json, 'r') as f:
        cam = json.load(f)
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    set_intrinsics_from_K_matrix(K, cam["width"], cam["height"])

    rendered = 0
    tries = 0
    while rendered < args.num_images and tries < 10000:
        print(f"\n--- Starting render attempt {tries+1}, completed {rendered} images ---")
        bproc.utility.reset_keyframes()

        tool_paths = random.sample(tools, k=random.choice([1, 2]))
        objs = []
        for tool_path in tool_paths:
            print(f"Loading tool: {tool_path}")
            loaded = bproc.loader.load_obj(tool_path)
            if not loaded:
                continue

            tool_obj = loaded[0]
            tool_obj.set_cp("category_id", 1)
            enforce_principled_shader(tool_obj)

            location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.5, radius_max=1.5)
            tool_obj.set_location(location)

            diag_rotation = Euler([
                random.uniform(np.radians(-45), np.radians(45)),
                random.uniform(np.radians(-45), np.radians(45)),
                random.uniform(np.radians(-180), np.radians(180))
            ])
            tool_obj.set_rotation_euler(diag_rotation)

            for mesh_obj in loaded:
                if hasattr(mesh_obj.blender_obj, "data") and hasattr(mesh_obj.blender_obj.data, "use_auto_smooth"):
                    mesh_obj.blender_obj.data.use_auto_smooth = True
                mesh_obj.blender_obj.show_transparent = False
                mesh_obj.blender_obj.cycles.use_transparent_shadow = False
            objs.extend(loaded)

        if not objs:
            tries += 1
            continue

        load_hdr_background(random.choice(hdrs))

        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=5))
        light.set_energy(random.uniform(150, 500))

        radii = [get_bounding_radius(o) for o in objs]
        max_radius = max(radii)
        half_fov_rad = bpy.data.cameras[0].angle / 2
        min_cam_distance = max_radius / np.sin(half_fov_rad)

        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=min_cam_distance + 0.5, radius_max=min_cam_distance + 1.5, elevation_min=-30, elevation_max=30)
        look_at = np.mean([o.get_location() for o in objs], axis=0)
        rotation = bproc.camera.rotation_from_forward_vec(look_at - location)
        cam2world = bproc.math.build_transformation_mat(location, rotation)

        if not all(is_fully_in_frame(obj, cam2world) for obj in objs):
            print("All pose attempts failed to capture all objects in frame.")
            tries += 1
            continue

        bproc.camera.add_camera_pose(cam2world)

        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.set_max_amount_of_samples(80)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        data = bproc.renderer.render()

        bproc.writer.write_coco_annotations(
            os.path.join(args.output_dir, "annotations"),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            append_to_existing_output=True,
            mask_encoding_format="rle"
        )

        image_path = os.path.join(args.output_dir, "images", f"{rendered:06d}.png")
        imageio.imwrite(image_path, (data["colors"][0] * 255).astype(np.uint8))

        for obj in bpy.data.objects:
            if obj.type == "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for mat in bpy.data.materials:
            bpy.data.materials.remove(mat, do_unlink=True)

        rendered += 1
        tries += 1

if __name__ == "__main__":
    main()