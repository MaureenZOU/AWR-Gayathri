import copy
import os
import os.path as osp

import bpy
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from mathutils import Vector
from regex import T
from sklearn.tree import export_text
from sympy import rad


# Load RigNet format
def load_lines(path: str, key: str) -> list:
    """Load lines from txt file that start with the key.

    Args:
        path (str): Path to the txt file
        key (str): Key to filter the lines

    Returns:
        list: List of lines that start with the key
    """
    # Read the txt file
    with open(path, "r") as f:
        lines = f.readlines()
    # Read lines start with 'skin'
    infos = []
    for line in lines:
        if line.startswith(key):
            infos.append(line.split("\n")[0])
    return infos


def load_joints(path: str, max_joints_num: int, joints_dict: dict) -> np.ndarray:
    """Load joints from the rig_info file.

    Args:
        path (str): Path to the rig_info file
        max_joints_num (int): Max joints number
        joints_dict (dict): Dictionary of joints

    Returns:
        np.ndarray: Joints position
    """
    joints_pos = np.zeros((max_joints_num, 4))
    joints_list = load_lines(path, "joint")
    for i, joint in enumerate(joints_list):
        joint = joint.split(" ")
        pos = np.array([float(joint[2]), float(joint[3]), float(joint[4])])
        name = joint[1]
        joint_idx = joints_dict[name]
        joints_pos[joint_idx] = np.array(
            [float(joint[2]), float(joint[3]), float(joint[4]), 1.0]
        )
    return joints_pos


def load_skin(
    pc_path: str, path: str, max_joints_num: int, joint_dict: dict
) -> np.ndarray:
    """Load skinning weights from the rig_info file.

    Args:
        pc_path (str): Path to the original mesh file
        path (str): Path to the rig_info file
        pointcloud (np.ndarray): Sampled pointcloud
        max_joints_num (int): Max joints number
        joint_dict (dict): Dictionary of joints

    Returns:
        np.ndarray: [vert_num, joints_num] Skinning weights
        np.ndarray: [point_num, joints_num] Interpolated skinning weights
    """
    skin_list = load_lines(path, "skin")

    # Load the original mesh and skip if the original data is invalid
    mesh = trimesh.load(pc_path)
    vertices = np.asarray(mesh.vertices)
    vert_num = vertices.shape[0]
    if len(skin_list) != vert_num:
        print(
            f"Error in skinning weights, the number of weights is not equal to the number of vertices."
        )
        return None, None
    assert (
        len(skin_list) == vert_num
    ), "The number of skinning weights is not equal to the number of vertices."

    # Load skinning weights
    skinning_weights = np.zeros((vert_num, max_joints_num))
    for i, skin in enumerate(skin_list):
        skin = skin.split(" ")
        assert (len(skin) - 3) % 2 == 0, "The number of linked joints has error."
        linked_joints_num = (len(skin) - 3) // 2
        for j in range(linked_joints_num):
            joint_name = skin[2 + 2 * j]
            weight = float(skin[2 + 1 + 2 * j])
            if joint_name not in joint_dict:
                print(f"Joint {joint_name} is not in the joint dictionary.")
                return None
            joint_pos = joint_dict[joint_name]
            assert (
                joint_pos is not None
            ), f"Joint {joint_name} is not in the joint dictionary."
            joint_idx = joint_dict[f"{joint_name}"]
            skinning_weights[i, joint_idx] = weight

    return skinning_weights


def load_bones(path: str, max_joints_num: int, joint_dict: dict) -> np.ndarray:
    """Load bones from the rig_info file.

    Args:
        path (str): Path to the rig_info file
        max_joints_num (int): Max joints number
        joint_dict (dict): Joint dictionary

    Returns:
        np.ndarray: [max_joints_num, max_joints_num] A connectivity matrix of bones
    """
    bones_list = load_lines(path, "hier")
    bones = np.zeros((max_joints_num, max_joints_num))
    for bone in bones_list:
        bone = bone.split(" ")
        joint1 = bone[1]
        joint2 = bone[2]
        assert joint1 in joint_dict, f"Joint {joint1} is not in the joint dictionary."
        assert joint2 in joint_dict, f"Joint {joint2} is not in the joint dictionary."
        joint1_idx = joint_dict[f"{joint1}_idx"]
        joint2_idx = joint_dict[f"{joint2}_idx"]
        bones[joint1_idx, joint2_idx] = 1
    return bones


def load_parents(path: str, max_joints_num: int) -> tuple[np.ndarray, dict]:
    """Load parents from the rig_info file.

    Args:
        path (str): Path to the rig_info file
        max_joints_num (int): Max joints number
    """
    # Load the root joint name
    root_list = load_lines(path, "root")
    for r in root_list:
        r = r.split(" ")
        root_joint_name = r[1]
        root_idx = 0

    # Store the joint name and index
    joint_dict = {}
    parent_list = load_lines(path, "hier")
    parents = np.arange(max_joints_num)
    curr_idx = root_idx
    for parent in parent_list:
        parent = parent.split(" ")
        parent_joint = parent[1]
        joint = parent[2]

        # Add parent idx to the joint_dict
        if parent_joint not in joint_dict.keys():
            joint_dict[parent_joint] = curr_idx
            parents[joint_dict[parent_joint]] = joint_dict[parent_joint]
            curr_idx += 1

        # Add joint idx to the joint_dict
        assert (
            joint not in joint_dict.keys()
        ), f"Joint {joint} is already in the joint dictionary."
        joint_dict[joint] = curr_idx
        curr_idx += 1

        # Store the parent idx
        parents[joint_dict[joint]] = joint_dict[parent_joint]

    return parents, joint_dict


def setup_view_layers():
    """Create and setup separate view layers for mesh and skeleton"""
    # Create new view layer for skeleton
    skeleton_layer = bpy.context.scene.view_layers.new("Skeleton")
    mesh_layer = bpy.context.scene.view_layers.new("Mesh")
    # mesh_layer = bpy.context.view_layer  # Use default view layer for mesh

    # Set up collections
    skeleton_collection = bpy.data.collections.new("Skeleton_Collection")
    mesh_collection = bpy.data.collections.new("Mesh_Collection")

    bpy.context.scene.collection.children.link(skeleton_collection)
    bpy.context.scene.collection.children.link(mesh_collection)

    # Set up view layer collection visibility
    skeleton_layer.layer_collection.children["Skeleton_Collection"].exclude = False
    skeleton_layer.layer_collection.children["Mesh_Collection"].exclude = True

    mesh_layer.layer_collection.children["Skeleton_Collection"].exclude = True
    mesh_layer.layer_collection.children["Mesh_Collection"].exclude = False

    return skeleton_collection, mesh_collection


def setup_compositor():
    """Setup compositor nodes for combining view layers"""
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes
    tree.nodes.clear()

    # Create input nodes for both render layers
    rl_skeleton = tree.nodes.new("CompositorNodeRLayers")
    rl_skeleton.layer = "Skeleton"
    rl_skeleton.location = (-200, 200)

    rl_mesh = tree.nodes.new("CompositorNodeRLayers")
    rl_mesh.layer = bpy.context.view_layer.name
    rl_mesh.location = (-200, -200)

    # Create Alpha Over node
    alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (200, 0)

    # Create output node
    composite_out = tree.nodes.new("CompositorNodeComposite")
    composite_out.location = (400, 0)

    # Link nodes
    tree.links.new(rl_skeleton.outputs["Image"], alpha_over.inputs[2])
    tree.links.new(rl_mesh.outputs["Image"], alpha_over.inputs[1])
    tree.links.new(alpha_over.outputs[0], composite_out.inputs[0])


def create_joint_sphere(location, radius=0.022, is_root=False, collection=None):
    """Create a sphere mesh at the specified joint location"""
    # Create sphere mesh
    sphere_mesh = bpy.data.meshes.new("Joint_Sphere")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    temp_sphere = bpy.context.active_object
    sphere_mesh.from_pydata(
        [
            (v.co.x + location[0], v.co.y + location[1], v.co.z + location[2])
            for v in temp_sphere.data.vertices
        ],
        [],
        [p.vertices for p in temp_sphere.data.polygons],
    )
    bpy.data.objects.remove(temp_sphere, do_unlink=True)

    # Create sphere object
    sphere = bpy.data.objects.new("Joint_Sphere", sphere_mesh)

    # Add to collection
    if collection:
        collection.objects.link(sphere)
    else:
        bpy.context.scene.collection.objects.link(sphere)

    # Smooth the sphere
    for poly in sphere.data.polygons:
        poly.use_smooth = True

    # Create and assign material
    mat = bpy.data.materials.new(
        name=f"Joint_{'Root' if is_root else 'Regular'}_Material"
    )
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create emission node
    # emission = nodes.new(type='ShaderNodeEmission')
    # emission.inputs[0].default_value = (1, 0, 0, 1) if is_root else (0, 1, 0, 1)

    # Create BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (
        (1, 0, 0, 1) if is_root else (0, 0.8, 0.8, 1.0)
    )
    bsdf.inputs["Emission Color"].default_value = (
        (1, 0, 0, 1) if is_root else (0, 1, 0, 1)
    )
    bsdf.inputs["Emission Strength"].default_value = 0.0

    # Create material output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs[0], material_output.inputs[0])

    # Assign material to sphere
    sphere.data.materials.append(mat)

    return sphere


def create_bone_cone(start_pos, end_pos, base_radius=0.015, collection=None):
    """Create a cone between two joint positions to represent a bone"""
    # Calculate direction and length
    direction = Vector(end_pos) - Vector(start_pos)
    length = direction.length

    if length < 0.0001:  # Skip if joints are too close
        return None

    # Create cone mesh
    cone_mesh = bpy.data.meshes.new("Bone_Cone")
    bpy.ops.mesh.primitive_cone_add(radius1=base_radius, radius2=0, depth=length)
    temp_cone = bpy.context.active_object
    cone_mesh.from_pydata(
        [v.co for v in temp_cone.data.vertices],
        [],
        [p.vertices for p in temp_cone.data.polygons],
    )
    bpy.data.objects.remove(temp_cone, do_unlink=True)

    # Create cone object
    cone = bpy.data.objects.new("Bone_Cone", cone_mesh)

    # Add to collection
    if collection:
        collection.objects.link(cone)
    else:
        bpy.context.scene.collection.objects.link(cone)

    # Position and rotate the cone
    cone.location = Vector(end_pos) - direction / 2
    direction.normalize()
    rot_quat = direction.to_track_quat("Z", "Y")
    cone.rotation_euler = rot_quat.to_euler()

    # Create and assign blue material
    mat = bpy.data.materials.new(name="Bone_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create emission node
    # emission = nodes.new(type='ShaderNodeEmission')
    # emission.inputs[0].default_value = (0, 0, 1, 1)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0, 0, 1.0, 1)
    bsdf.inputs["Emission Color"].default_value = (0, 0, 1, 1)
    bsdf.inputs["Emission Strength"].default_value = 0.0

    # Create material output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs[0], material_output.inputs[0])

    # Assign material to cone
    cone.data.materials.append(mat)

    return cone


def assign_single_weight(obj, bone_name, weight=1.0):
    """Assign weight 1.0 to a single bone and 0.0 to others"""
    # First, create vertex groups for all bones
    for bone in bpy.data.objects["armature"].data.bones:
        group = obj.vertex_groups.new(name=bone.name)
        vertices = [v.index for v in obj.data.vertices]
        group.add(vertices, 0.0, "REPLACE")

    # Then assign weight 1.0 to the target bone
    target_group = obj.vertex_groups[bone_name]
    vertices = [v.index for v in obj.data.vertices]
    target_group.add(vertices, weight, "REPLACE")


def create_skeleton_visualization(
    joint_positions, joint_hierarchy, collection=None, joint_size=0.022, bone_size=0.015
):
    """Create visual representation of the skeleton with spheres and cones"""
    armature_obj = bpy.data.objects["armature"]

    # Create spheres for joints
    spheres = []
    for i, pos in enumerate(joint_positions):
        sphere = create_joint_sphere(
            pos[..., :3], is_root=(i == 0), collection=collection, radius=joint_size
        )
        sphere.name = f"Joint_Sphere_{i}"

        # Parent to armature
        # sphere.parent = armature_obj

        # Assign weight to corresponding bone
        assign_single_weight(sphere, f"Bone_{i}")

        # Add armature modifier
        modifier = sphere.modifiers.new(name="Armature", type="ARMATURE")
        modifier.object = armature_obj

        spheres.append(sphere)

    # Create cones for bones
    cones = []
    for i, pos in enumerate(joint_positions):
        if i == 0:  # Skip root
            continue

        parent_idx = joint_hierarchy[i]
        if parent_idx == i:  # Skip disconnected joints
            continue

        start_pos = joint_positions[parent_idx][..., :3]
        end_pos = pos[..., :3]

        cone = create_bone_cone(
            start_pos, end_pos, collection=collection, base_radius=bone_size
        )
        if cone is not None:
            cone.name = f"Bone_Cone_{i}"

            # Parent to armature
            # cone.parent = armature_obj

            # Assign weight to corresponding bone
            assign_single_weight(cone, f"Bone_{parent_idx}")

            # Add armature modifier
            modifier = cone.modifiers.new(name="Armature", type="ARMATURE")
            modifier.object = armature_obj

            cones.append(cone)

    return spheres, cones


def create_armature(joint_positions, joint_hierarchy):
    # Create armature and enter edit mode
    armature = bpy.data.armatures.new("armature")
    armature_obj = bpy.data.objects.new("armature", armature)
    bpy.context.scene.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    # Create bones
    edit_bones = armature.edit_bones
    for i, pos in enumerate(joint_positions):
        bone = edit_bones.new(f"Bone_{i}")
        bone.head = pos[..., :3]
        bone.tail = pos[..., :3] + [0, 0, 0.1]  # Arbitrary bone length

        if joint_hierarchy[i] == i and i > 0:  # for non-exists joints
            continue
        if i > 0:
            parent = edit_bones[f"Bone_{joint_hierarchy[i]}"]
            bone.parent = parent
            parent.tail = bone.head
        # elif i == 0:  # For root bone, add small offset to tail
        #     bone.tail = [pos[0], pos[1], pos[2] + 0.1]  # Arbitrary small offset for root bone

    bpy.ops.object.mode_set(mode="OBJECT")

    return armature_obj


def point_cloud_to_mesh(points, weights, mesh):
    """Convert point cloud to mesh with weights"""
    # Create Blender mesh
    blender_mesh = bpy.data.meshes.new("PointCloudMesh")
    blender_mesh.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
    blender_mesh.update()

    vertex_weights = []
    # if points is not None:
    #     # Interpolate weights to vertices
    #     for vertex in mesh.vertices:
    #         # Find k-nearest points in original point cloud
    #         k = 1
    #         distances = np.linalg.norm(points - vertex, axis=1)
    #         nearest_indices = np.argsort(distances)[:k]
    #         nearest_weights = weights[nearest_indices]
    #         vertex_weights.append(nearest_weights.mean(axis=0))
    # else:
    #     for idx, vertex in enumerate(mesh.vertices):
    #         vertex_weights.append(weights[idx])
    for idx, vertex in enumerate(mesh.vertices):
        vertex_weights.append(weights[idx])
    return blender_mesh, vertex_weights


def create_mesh_object(mesh, mesh_collection=None):
    """Create mesh object and link to scene"""
    obj = bpy.data.objects.new("PointCloudObject", mesh)
    if mesh_collection:
        mesh_collection.objects.link(obj)
    else:
        bpy.context.scene.collection.objects.link(obj)

    # Create material
    mat = bpy.data.materials.new(name="WhiteMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    nodes.clear()

    # Create a principled BSDF node
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (
        0.566,
        0.862,
        1.0,
        1,
    )  # White color
    principled.inputs["Metallic"].default_value = 0.02

    # Create material output node
    material_output = nodes.new("ShaderNodeOutputMaterial")

    # Link nodes
    mat.node_tree.links.new(
        principled.outputs["BSDF"], material_output.inputs["Surface"]
    )

    # Assign material to object
    obj.data.materials.append(mat)

    # Enable wireframe display
    # obj.show_wire = True  # Show wireframe

    return obj


def assign_weights(obj, armature, vertex_weights):
    """Assign weights to mesh vertices"""
    # Assign armature modifier
    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature

    # Assign vertex groups and weights
    for i, bone in enumerate(armature.data.bones):
        bone_idx = int(bone.name.split("_")[-1])
        group = obj.vertex_groups.new(name=bone.name)
        for j, weights in enumerate(vertex_weights):
            group.add([j], weights[bone_idx], "REPLACE")


def main(
    points, joint_positions, joint_hierarchy, skinning_weights, mesh_orig, save_path
):
    """Main function to create and export the model"""
    # Set up rendering engine
    bpy.context.scene.render.engine = "CYCLES"
    # Use GPU
    bpy.context.scene.cycles.device = "GPU"
    # Change color management to standard
    bpy.context.scene.view_settings.view_transform = "Standard"

    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Setup view layers and collections
    skeleton_collection, mesh_collection = setup_view_layers()

    # # Create armature
    armature = create_armature(joint_positions, joint_hierarchy)

    # # Create mesh from point cloud
    mesh, vertex_weights = point_cloud_to_mesh(points, skinning_weights, mesh_orig)

    # Create mesh object
    mesh_obj = create_mesh_object(mesh, mesh_collection)

    # # Assign weights
    assign_weights(mesh_obj, armature, vertex_weights)

    # Clear custom shapes
    for pose_bone in armature.pose.bones:
        pose_bone.custom_shape = None

    # Decide the joint and bone size
    expand = mesh_orig.vertices.max() - mesh_orig.vertices.min()
    expand = expand.min()
    joint_size = expand * 0.008
    bone_size = joint_size * 0.8

    # Add skeleton visualization to skeleton collection
    spheres, cones = create_skeleton_visualization(
        joint_positions,
        joint_hierarchy,
        collection=skeleton_collection,
        joint_size=joint_size,
        bone_size=bone_size,
    )

    # Set parents to mesh, armature, and skeleton visualization
    mesh_obj.parent = armature
    for sphere in spheres:
        sphere.parent = armature
    for cone in cones:
        cone.parent = armature

    # Setup compositor
    setup_compositor()

    # Add a plane as a shadow catcher
    plane_mesh = bpy.ops.mesh.primitive_plane_add(
        size=100,
        enter_editmode=False,
        align="WORLD",
        location=(0, 0, 0),
        scale=(1, 1, 1),
    )
    plane_obj = bpy.context.active_object
    plane_obj.name = "Plane"
    plane_obj.is_shadow_catcher = True
    mesh_collection.objects.link(plane_obj)

    # Set up camera if needed
    if not bpy.context.scene.camera:
        # bpy.ops.object.camera_add(location=(3.0, -3.0, 1), rotation=(0, 0, 0))
        bpy.ops.object.camera_add(
            location=(3.0, -3.0, 1.2), rotation=(0, 0, 0)
        )  # for real
        # bpy.ops.object.camera_add(location=(0.0, -3.0, 1), rotation=(0, 0, 0))
        camera = bpy.context.active_object
        bpy.context.scene.camera = camera
        # Set up camera contraint to track the empty obj
        empty_obj = bpy.data.objects.new("Empty", None)
        empty_obj.location = (0, 0, 0.5)
        bpy.context.scene.collection.objects.link(empty_obj)
        track_constraint = camera.constraints.new("TRACK_TO")
        track_constraint.target = empty_obj
        # Set up camera orthographic
        bpy.context.scene.camera.data.type = "ORTHO"
        bpy.context.scene.camera.data.ortho_scale = 3.0

    # Set up lighting for mesh layer
    mesh_light = bpy.data.lights.new(name="mesh_light", type="AREA")
    mesh_light_object = bpy.data.objects.new(name="mesh_light", object_data=mesh_light)
    mesh_light_object.location = (0.81316, -3.5065, 11.209)
    mesh_light_object.rotation_euler = (30.367, -11.459, 0)
    mesh_light.energy = 1800
    mesh_light.size = 10
    bpy.context.scene.collection.objects.link(mesh_light_object)
    track_constraint = mesh_light_object.constraints.new("TRACK_TO")
    track_constraint.target = empty_obj

    # Set up world background lighting
    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    node_tree = world.node_tree
    background_node = node_tree.nodes.get("Background")
    # Modify the background node's properties
    if background_node:
        background_node.inputs[0].default_value = (
            0.0,
            0.0,
            0.0,
            1.0,
        )  # Set color to red

    # set up film transparent
    bpy.context.scene.render.film_transparent = True

    # Export mesh glb
    ## Select spheres and cones
    bpy.context.window.view_layer = bpy.context.scene.view_layers["Mesh"]
    skel_glb_path = save_path.replace(".blend", "_rig.glb")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=skel_glb_path,
        export_format="GLB",  # Must be 'GLB' or 'GLTF_SEPARATE' or 'GLTF_EMBEDDED'
        use_selection=True,  # Must be True or False
        export_draco_mesh_compression_enable=True,  # Must be True or False
        # Other parameters...
        export_materials="EXPORT",  # Use string enum instead of bool
    )

    # Free memory
    import gc

    gc.collect()


# Load riganything format data
def load_riganything(data_path, mesh_orig, is_pred=True):
    mode = "pred" if is_pred else "gt"
    print(f"Loading RigAnything data in {mode} mode...")
    data = np.load(data_path, allow_pickle=True)
    points = data["pointcloud"]
    joint_positions = data["joints"][..., :3]  # [J, 3]
    # joint_positions = data['joints'][:, 0, :3]   # [J, 3]  # for ablation 1
    # print(joint_positions.shape)
    parent_indices = data["parents"]  # [J]
    skinning_weights = data["skinning_weights"]  # [N, J]

    # Delete the non-exist joints
    if mode == "gt":
        joint_masks = data["joints"][..., -1] == 1
        num_joints = int(joint_masks.sum())
        points = points[:, :3]
        joint_positions = joint_positions[:num_joints]
        parent_indices = parent_indices[:num_joints]
        skinning_weights = skinning_weights[:, :num_joints]

    skinning_weights /= (
        skinning_weights.sum(axis=1, keepdims=True) + 1e-6
    )  # Normalize weights

    # points = np.dot(points, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    # joint_positions = np.dot(joint_positions, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    # mesh_orig.vertices = np.dot(mesh_orig.vertices, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    return points, joint_positions, parent_indices, skinning_weights, mesh_orig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    args = parser.parse_args()

    # Load mesh
    mesh_orig = trimesh.load(args.mesh_path, force="mesh")
    (
        points,
        joint_positions,
        parent_indices,
        skinning_weights,
        mesh_orig,
    ) = load_riganything(args.data_path, mesh_orig, is_pred=True)

    os.makedirs(args.save_path, exist_ok=True)
    save_path = osp.join(
        args.save_path, osp.basename(args.data_path).split(".")[0] + f".blend"
    )

    main(
        points,
        joint_positions,
        parent_indices,
        skinning_weights,
        mesh_orig,
        save_path=save_path,
    )
