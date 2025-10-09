"""
 # @ Author: Isabella Liu
 # @ Create Time: 2024-12-04 18:45:04
 # @ Modified by: Isabella Liu
 # @ Modified time: 2024-08-01 18:45:14
 # @ Description: Process the RigNet dataset so that it can used in the LRM model, store the joint in a BFS order, only rotate along z-axis, add deformed pointcloud
 """

import glob
import os
import os.path as osp
import queue
import random
import sys
import threading

import numpy as np
import open3d as o3d
import trimesh
from networkx import barycenter
from scipy import interpolate
from scipy.datasets import face
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def load_pc(path: str, sample_num: int = 2048, sorting: bool = True) -> np.ndarray:
    """Load point cloud from file.

    Args:
        path (str): Path to the .obj file

    Returns:
        np.ndarray: Vertices of the mesh
    """
    mesh = trimesh.load(path)
    # Uniformly sample points from the mesh  TODO deal with the skinning weights
    # pc = mesh.sample_points_uniformly(sample_num)

    vert_points = np.asarray(mesh.vertices)
    vert_normals = np.asarray(mesh.vertex_normals)
    face_sample_num = sample_num - vert_points.shape[0]

    if face_sample_num <= 0:
        # random sample points
        idx = np.random.choice(vert_points.shape[0], sample_num, replace=False)
        pc = vert_points[idx]
        normals = vert_normals[idx]
        return vert_points, pc, normals, idx
    else:
        # Uniformly sample points and its normals from the mesh
        sampled_points, face_indices = trimesh.sample.sample_surface_even(
            mesh, face_sample_num
        )
        sampled_normals = mesh.face_normals[face_indices]
        sampled_points = np.asarray(sampled_points)
        sampled_normals = np.asarray(sampled_normals)

        pc = np.concatenate((vert_points, sampled_points), axis=0)
        normals = np.concatenate((vert_normals, sampled_normals), axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        if sorting:
            # Sorting the pointcloud in x,y,z order
            pc_sort_idx = np.lexsort((pc[:, 0], pc[:, 1], pc[:, 2]))
            pc = pc[pc_sort_idx]
            normals = normals[pc_sort_idx]

        pc_old = np.asarray(mesh.vertices)

        return pc_old, pc, normals, None


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
    pc_path: str,
    path: str,
    pointcloud: int,
    max_joints_num: int,
    joint_dict: dict,
    pc_idx: np.array = None,
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

    # Interpolate the skinning weight on the sampled pointcloud
    interpolate_num = pointcloud.shape[0] - vert_num
    if interpolate_num <= 0:
        # Normalize the skinning weights
        skinning_weights = skinning_weights / (
            np.sum(skinning_weights, axis=1, keepdims=True) + 1e-6
        )
        return skinning_weights, skinning_weights[pc_idx]
    else:
        sampled_points = pointcloud[vert_num:]
        interpolated_skinning_weights = np.zeros((interpolate_num, max_joints_num))
        ## Find which faces the points are in
        closest_faces = mesh.nearest.on_surface(sampled_points)[
            2
        ]  # [sampeld_point_num]
        face_vertices = mesh.vertices[
            mesh.faces[closest_faces]
        ]  # [sampeld_point_num, 3, 3]
        # Compute the baycentric coordinates
        barycentric_coords = trimesh.triangles.points_to_barycentric(
            face_vertices, sampled_points
        )  # [sampeld_point_num, 3]
        interpolated_skinning_weights = np.sum(
            barycentric_coords[:, :, None]
            * skinning_weights[mesh.faces[closest_faces]],
            axis=1,
        )  # [sampeld_point_num, joints_num]

        # Check for validation, the sum of weights should be 1
        # if not np.allclose(np.sum(interpolated_skinning_weights, axis=1), 1):
        #     print(f"Error in skinning weights, the sum of skinning weights is not 1.")
        #     return None, None
        # assert np.allclose(np.sum(interpolated_skinning_weights, axis=1), 1), "The sum of skinning weights is not 1."

        # Concatenate the skinning weights
        final_skinning_weights = np.concatenate(
            (skinning_weights, interpolated_skinning_weights), axis=0
        )
        # Normalize the skinning weights
        final_skinning_weights = final_skinning_weights / (
            np.sum(final_skinning_weights, axis=1, keepdims=True) + 1e-6
        )

        return skinning_weights, final_skinning_weights


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


def load_root(path: str, joint_dict: dict) -> np.ndarray:
    """Load root joint index from the rig_info file.

    Args:
        path (str): Path to the rig_info file
        joint_dict (dict): Joint dictionary

    Returns:
        np.ndarray: [1] Root joint index
    """
    root_list = load_lines(path, "root")
    root_idx = np.zeros(1)
    assert len(root_list) == 1, "There are more than one root in the rig_info file."
    for r in root_list:
        r = r.split(" ")
        assert r[1] in joint_dict, f"Joint {r[1]} is not in the joint dictionary."
        root_idx[0] = joint_dict[r[1]]
    return root_idx


def get_max_joints_num(path: str) -> int:
    """Get the max joints number in the dataset.

    Args:
        path (str): Path to the folder that contains all the rig_info files

    Returns:
        int: Max joints number
    """
    print("Getting max joints number...")
    # Get all files under the folder
    files = glob.glob(osp.join(path, "*.txt"))
    # Get max joints number
    max_joints_num = 0
    for file in tqdm(files):
        joints = load_lines(file, "joint")
        max_joints_num = max(max_joints_num, len(joints))
    print(f"Max joints number: {max_joints_num}")
    return max_joints_num


def process_file(
    file_queue,
    output_folder,
    max_joints_num,
    sort_pointcloud_xyz,
    ROT_TIMES,
    pc_folder,
    rig_info_folder,
):
    while True:
        try:
            file = file_queue.get_nowait()
        except queue.Empty:
            break

        item_idx = file.split("/")[-1].split(".")[0]
        pc_path = osp.join(pc_folder, f"{item_idx}.obj")
        rig_info_path = osp.join(rig_info_folder, f"{item_idx}.txt")

        # Process infos
        pointcloud_old, pointcloud, normals, pc_idx = load_pc(
            pc_path, sorting=sort_pointcloud_xyz
        )
        parents, joints_dict = load_parents(rig_info_path, max_joints_num)
        joints = load_joints(rig_info_path, max_joints_num, joints_dict)
        skinning_weights, interpolated_skinning_weights = load_skin(
            pc_path, rig_info_path, pointcloud, max_joints_num, joints_dict, pc_idx
        )
        if skinning_weights is None:
            print(f"Skip {item_idx} because of error in skinning weights.")
            file_queue.task_done()
            continue
        root_idx = load_root(rig_info_path, joints_dict)

        # Compose infos to dict with original pointcloud
        item_dict = {
            "pointcloud": pointcloud,
            "normals": normals,
            "joints": joints,
            "parents": parents,
            "joints_dict": joints_dict,
            "skinning_weights": interpolated_skinning_weights,
            "root_idx": root_idx,
        }
        # Save dict to npz file
        output_path = osp.join(output_folder, f"npz/{item_idx}.npz")
        np.savez(output_path, **item_dict)

        # Data augmentation
        for rot_idx in range(ROT_TIMES):
            random_rotation = R.from_euler("y", random.uniform(0, 360), degrees=True)
            rotation_matrix = random_rotation.as_matrix()

            rotated_pointcloud = (rotation_matrix @ pointcloud.T).T
            rotated_normals = (rotation_matrix @ normals.T).T
            rotated_joints = joints.copy()
            rotated_joints[:, :3] = (rotation_matrix @ joints[:, :3].T).T

            rotated_item_dict = {
                "pointcloud": rotated_pointcloud,
                "normals": rotated_normals,
                "joints": rotated_joints,
                "parents": parents,
                "joints_dict": joints_dict,
                "skinning_weights": interpolated_skinning_weights,
                "root_idx": root_idx,
            }
            output_path = osp.join(output_folder, f"npz/{item_idx}_r_{rot_idx}.npz")
            np.savez(output_path, **rotated_item_dict)

        file_queue.task_done()


def process_split_files(output_folder, ROT_TIMES):
    # Get the list of processed files
    processed_files = set(
        f.split(".")[0] for f in os.listdir(osp.join(output_folder, "npz"))
    )

    for split in ["train", "test", "val"]:
        input_file = f"/sensei-fs/users/lanxinl/data/ModelResource_RigNetv1_preproccessed/{split}_final.txt"
        output_file = osp.join(output_folder, f"{split}_final.txt")

        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                item = line.strip()
                if item in processed_files:
                    f_out.write(f"{item}\n")
                    for rot_idx in range(ROT_TIMES):
                        f_out.write(f"{item}_r_{rot_idx}\n")


def main():
    rig_info_folder = "/sensei-fs/users/lanxinl/data/ModelResource_RigNetv1_preproccessed/rig_info_remesh"
    # rig_info_folder = "/sensei-fs/users/lanxinl/data/ModelResource_RigNetv1_preproccessed/rig_info"
    pc_folder = (
        "/sensei-fs/users/lanxinl/data/ModelResource_RigNetv1_preproccessed/obj_remesh"
    )
    # pc_folder = "/sensei-fs/users/lanxinl/data/ModelResource_RigNetv1_preproccessed/obj"
    output_folder = "/sensei-fs/users/lanxinl/data/RigLRM_data_12_25"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(osp.join(output_folder, "npz"), exist_ok=True)

    max_joints_num = 64
    sort_pointcloud_xyz = False
    ROT_TIMES = 0

    files = glob.glob(osp.join(rig_info_folder, '*.txt'))
    file_queue = queue.Queue()
    for file in files:
        file_queue.put(file)

    # num_threads = os.cpu_count()  # Use the number of CPU cores
    num_threads = 80  # Use the number of CPU cores
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=process_file, args=(file_queue, output_folder, max_joints_num, sort_pointcloud_xyz, ROT_TIMES, pc_folder, rig_info_folder))
        thread.start()
        threads.append(thread)

    # Wait for all files to be processed
    file_queue.join()

    # Stop the threads
    for _ in range(num_threads):
        file_queue.put(None)
    for thread in threads:
        thread.join()

    # Process the split files
    process_split_files(output_folder, ROT_TIMES)


if __name__ == "__main__":
    main()
