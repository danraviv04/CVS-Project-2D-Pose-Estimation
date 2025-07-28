# import blenderproc as bproc
# from blenderproc.python.camera import CameraUtility
# import bpy
# import numpy as np
# import argparse
# import random
# import os
# import json
# import glob
# from mathutils import Matrix

# parser = argparse.ArgumentParser()
# parser.add_argument('--obj_dir', default="/datashare/project/surgical_tools_models", help="Directory containing .obj files")
# parser.add_argument('--camera_params', default="/datashare/project/camera.json", help="Camera intrinsics in json format")
# parser.add_argument('--output_dir', default="/home/student/project/output", help="Path to where the final files will be saved")
# parser.add_argument('--num_images', type=int, default=5, help="Number of images to generate")
# args = parser.parse_args()

# bproc.init()

# # Load all .obj tools in directory
# obj_files = glob.glob(os.path.join(args.obj_dir, "*/*.obj"))
# if len(obj_files) == 0:
#     raise RuntimeError("No .obj files found in directory.")

# with open(args.camera_params, "r") as file:
#     camera_params = json.load(file)

# K = np.array([[camera_params["fx"], 0, camera_params["cx"]],
#               [0, camera_params["fy"], camera_params["cy"]],
#               [0, 0, 1]])
# CameraUtility.set_intrinsics_from_K_matrix(K, camera_params["width"], camera_params["height"])

# # Prepare rendering settings
# bproc.renderer.set_max_amount_of_samples(100)
# bproc.renderer.set_output_format(enable_transparency=True)
# bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# for img_idx in range(args.num_images):
#     bproc.utility.reset_keyframes()

#     for obj in bpy.data.objects:
#         bpy.data.objects.remove(obj, do_unlink=True)

#     bproc.utility.reset_keyframes()

#     num_tools = random.choice([1, 2])
#     chosen_objs = random.sample(obj_files, k=num_tools)
#     loaded_objs = []

#     for obj_path in chosen_objs:
#         objs = bproc.loader.load_obj(obj_path)
#         obj = objs[0]
#         obj.set_cp("category_id", 1)
#         mat = obj.get_materials()[0] if obj.get_materials() else None
#         if mat:
#             mat.set_principled_shader_value("Specular IOR Level", random.uniform(0, 1))
#             mat.set_principled_shader_value("Roughness", 0.2)
#             mat.set_principled_shader_value("Metallic", 1)
#         # Random position and orientation
#         location = np.array([random.uniform(-0.2, 0.2), 0, random.uniform(-0.2, 0.2)])
        
#         rotation_np = bproc.sampler.uniformSO3()
#         rotation_bpy = Matrix(rotation_np.tolist())
#         obj.set_rotation_mat(rotation_bpy)

#         obj.set_location(location)
#         loaded_objs.append(obj)

#     # Lighting
#     light = bproc.types.Light()
#     light.set_type("POINT")
#     light.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=1.5, radius_max=3.0, elevation_min=30, elevation_max=85))
#     light.set_energy(random.uniform(500, 1500))

#     # Sample camera pose
#     tries = 0
#     while tries < 200:
#         cam_location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1.5, radius_max=3.0, elevation_min=-30, elevation_max=60)
#         look_at = np.array([0, 0, 0]) + np.random.uniform(-0.2, 0.2, size=3)
#         rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location, inplane_rot=random.uniform(-0.3, 0.3))
#         cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
#         if all([obj in bproc.camera.visible_objects(cam2world) for obj in loaded_objs]):
#             bproc.camera.add_camera_pose(cam2world)
#             break
#         tries += 1
#     else:
#         print(f"❌ Could not find visible pose for image {img_idx}. Skipping.")
#         continue

#     data = bproc.renderer.render()
#     instance_segmaps = data["instance_segmaps"]
#     for i, mask in enumerate(instance_segmaps):
#         if not isinstance(mask, np.ndarray):
#             raise RuntimeError(f"Segmap {i} is not a NumPy array")
#         if mask.dtype != np.uint8:
#             instance_segmaps[i] = (mask > 0).astype(np.uint8)

#     bproc.writer.write_coco_annotations(
#         os.path.join(args.output_dir, 'coco_data'),
#         instance_segmaps=instance_segmaps,
#         instance_attribute_maps=data["instance_attribute_maps"],
#         colors=data["colors"],
#         mask_encoding_format="rle",
#         append_to_existing_output=True
#     )

#     print(f"✅ Rendered image {img_idx + 1}/{args.num_images}")

import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse
import random
import os
import json
import glob
from mathutils import Matrix, Euler

parser = argparse.ArgumentParser()
parser.add_argument('--obj_dir', default="/datashare/project/surgical_tools_models", help="Directory containing .obj files")
parser.add_argument('--camera_params', default="/datashare/project/camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="/home/student/project/output", help="Path to where the final files will be saved")
parser.add_argument('--num_images', type=int, default=5, help="Number of images to generate")
args = parser.parse_args()

bproc.init()

# Ensure a Blender camera exists and is set as the active scene camera
if bpy.context.scene.camera is None:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

# Load all .obj tools in directory
obj_files = glob.glob(os.path.join(args.obj_dir, "*/*.obj"))
if len(obj_files) == 0:
    raise RuntimeError("No .obj files found in directory.")

# Load camera intrinsics
with open(args.camera_params, "r") as file:
    camera_params = json.load(file)

K = np.array([[camera_params["fx"], 0, camera_params["cx"]],
              [0, camera_params["fy"], camera_params["cy"]],
              [0, 0, 1]])
CameraUtility.set_intrinsics_from_K_matrix(K, camera_params["width"], camera_params["height"])

# Prepare rendering settings
bproc.renderer.set_max_amount_of_samples(100)
bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# Generate images
for img_idx in range(args.num_images):
    bproc.utility.reset_keyframes()

    for obj in bpy.data.objects:
        if obj.type != 'CAMERA':  # ✅ Keep the camera!
            bpy.data.objects.remove(obj, do_unlink=True)

    num_tools = random.choice([1, 2])
    chosen_objs = random.sample(obj_files, k=num_tools)
    loaded_objs = []

    for obj_path in chosen_objs:
        objs = bproc.loader.load_obj(obj_path)
        obj = objs[0]
        obj.set_cp("category_id", 1)

        mat = obj.get_materials()[0] if obj.get_materials() else None
        if mat:
            mat.set_principled_shader_value("Specular IOR Level", random.uniform(0, 1))
            mat.set_principled_shader_value("Roughness", 0.2)
            mat.set_principled_shader_value("Metallic", 1)

        # Random position and orientation
        location = np.array([random.uniform(-0.2, 0.2), 0, random.uniform(-0.2, 0.2)])
        rotation_euler = Euler([random.uniform(0, 2*np.pi) for _ in range(3)], 'XYZ')
        obj.set_rotation_euler(rotation_euler)

        obj.set_location(location)
        
        scale_factor = random.uniform(0.25, 0.45)
        obj.set_scale([scale_factor] * 3)

        
        loaded_objs.append(obj)

    # Add lighting
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=1.5, radius_max=3.0, elevation_min=30, elevation_max=85))
    light.set_energy(random.uniform(500, 1500))

    # Sample camera pose
    tries = 0
    while tries < 200:
        cam_location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1.5, radius_max=3.0, elevation_min=-30, elevation_max=60)
        look_at = np.array([0, 0, 0]) + np.random.uniform(-0.2, 0.2, size=3)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location, inplane_rot=random.uniform(-0.3, 0.3))
        cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)

        if all([obj in bproc.camera.visible_objects(cam2world) for obj in loaded_objs]):
            bproc.camera.add_camera_pose(cam2world)
            break
        tries += 1
    else:
        print(f"❌ Could not find visible pose for image {img_idx}. Skipping.")
        continue

    # Render and post-process
    data = bproc.renderer.render()
    instance_segmaps = data["instance_segmaps"]
    for i, mask in enumerate(instance_segmaps):
        if not isinstance(mask, np.ndarray):
            raise RuntimeError(f"Segmap {i} is not a NumPy array")
        if mask.dtype != np.uint8:
            instance_segmaps[i] = (mask > 0).astype(np.uint8)

    # Save COCO annotations
    bproc.writer.write_coco_annotations(
        os.path.join(args.output_dir, 'coco_data'),
        instance_segmaps=instance_segmaps,
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        mask_encoding_format="rle",
        append_to_existing_output=True
    )

    print(f"✅ Rendered image {img_idx + 1}/{args.num_images}")