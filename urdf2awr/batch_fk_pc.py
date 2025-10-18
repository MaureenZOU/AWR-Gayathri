import os
import json
import numpy as np
import pybullet as p
import open3d as o3d
from pathlib import Path
import torch
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ========= CONFIG =========
URDF_FILE = "/disk1/data/awr/robot/xhand_right/urdf/xhand_right.urdf"
JSON_DIR = "/disk1/data/awr/trellis/xhandr/joints_1/"
MESH_DIR = "/disk1/data/awr/robot/xhand_right/meshes"

# Voxelizer settings
VOXEL_RES = 512
MIN_BOUND = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
MAX_BOUND = np.array([ 0.5,  0.5,  0.5], dtype=np.float64)
EPS = 1e-6

# outputs
OUT_DIR = "/disk1/data/awr/trellis/xhandr/joints_1_data"

# ========= DEBUGGING =========
SAVE_INTERMEDIATE_MESH = False
INTERMEDIATE_MESH_FILE = "debug_combined_mesh.ply"
CONVERT_Z_UP_TO_Y_UP = True
# ==========================

@contextmanager
def suppress_c_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull
    at the C-level, silencing compiled libraries."""
    # Open a file to /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Duplicate the original stdout and stderr file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    # Redirect C-level stdout and stderr to devnull
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        # Restore the original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)

        # Close the duplicated file descriptors
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        # Close the devnull file descriptor
        os.close(devnull_fd)

def load_robot_and_set_joints(urdf_path, json_path):
    # Use the context manager to hide the print statements from loadURDF
    if not p.isConnected():
        client_id = p.connect(p.DIRECT)
    else:
        p.disconnect()
        client_id = p.connect(p.DIRECT)

    with suppress_c_stdout_stderr():
        urdf_flags = p.URDF_MERGE_FIXED_LINKS
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=urdf_flags)

    with open(json_path, "r") as f:
        joint_states = json.load(f)

    num_joints = p.getNumJoints(robot_id)
    joint_indices = {p.getJointInfo(robot_id, i)[1].decode("utf-8"): i for i in range(num_joints)}

    for jname, jval in joint_states.items():
        if jname in joint_indices:
            p.resetJointState(robot_id, joint_indices[jname], float(jval))

    p.stepSimulation()
    return robot_id, client_id


def T_from_pos_quat(pos, quat_xyzw):
    R_flat = p.getMatrixFromQuaternion(quat_xyzw)
    R = np.array(R_flat, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T

def gather_link_data(robot_id):
    """Gathers position, orientation (quat & matrix), and transform for each link."""
    link_data = {}

    # Base link (index -1)
    base_pos, base_quat_xyzw = p.getBasePositionAndOrientation(robot_id)
    link_data[-1] = {
        'pos': np.array(base_pos, dtype=np.float64),
        'quat': np.array(base_quat_xyzw, dtype=np.float64),
        'rotation': np.array(p.getMatrixFromQuaternion(base_quat_xyzw), dtype=np.float64).reshape(3, 3),
        'T': T_from_pos_quat(base_pos, base_quat_xyzw)
    }

    num_joints = p.getNumJoints(robot_id)
    for link_idx in range(num_joints):
        # computeForwardKinematics=True ensures the state is up-to-date
        ls = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
        link_pos = ls[4]
        link_quat_xyzw = ls[5]

        link_data[link_idx] = {
            'pos': np.array(link_pos, dtype=np.float64),
            'quat': np.array(link_quat_xyzw, dtype=np.float64),
            'rotation': np.array(p.getMatrixFromQuaternion(link_quat_xyzw), dtype=np.float64).reshape(3, 3),
            'T': T_from_pos_quat(link_pos, link_quat_xyzw)
        }
    return link_data

def get_mesh_scale_from_visual_tuple(vrec):
    if len(vrec) > 9 and isinstance(vrec[9], (list, tuple)) and len(vrec[9]) == 3:
        return np.array(vrec[9], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0], dtype=np.float64)

def main():
    json_files = sorted(list(Path(JSON_DIR).rglob('joint_states_*.json')))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not json_files:
        print(f"[error] No 'joint_states_*.json' files found in {JSON_DIR}. Please check the path.")
        return

    print(f"[info] Found {len(json_files)} joint files to process.")

    for json_path in tqdm(json_files, desc="Processing joint files"):
        with open(json_path, "r") as f:
            joint_states_to_save = json.load(f)

        robot_id, client_id = load_robot_and_set_joints(URDF_FILE, json_path)
        link_data = gather_link_data(robot_id)

        vis_data = p.getVisualShapeData(robot_id)
        all_world_meshes = []

        for vrec in vis_data:
            link_idx, geom_type, filename_bytes = vrec[1], vrec[2], vrec[4]
            v_local_pos, v_local_quat = vrec[5], vrec[6]

            if geom_type != p.GEOM_MESH or not filename_bytes:
                continue

            raw_filename = filename_bytes.decode('utf-8')
            base_filename = os.path.basename(raw_filename)
            resolved_filename = os.path.join(MESH_DIR, base_filename)

            if not os.path.exists(resolved_filename):
                # MODIFIED: Use tqdm.write for cleaner output with the progress bar
                tqdm.write(f"[warn] Skipping mesh, file not found: {resolved_filename}")
                continue

            mesh = o3d.io.read_triangle_mesh(resolved_filename)
            if mesh.is_empty():
                continue

            mesh_scale = get_mesh_scale_from_visual_tuple(vrec)
            mesh.scale(mesh_scale[0], center=(0, 0, 0))

            T_link_world = link_data[link_idx]['T']
            T_visual_link = T_from_pos_quat(v_local_pos, v_local_quat)
            T_visual_world = T_link_world @ T_visual_link
            mesh.transform(T_visual_world)

            all_world_meshes.append(mesh)

        if not all_world_meshes:
            tqdm.write(f"[error] No visual meshes were loaded successfully for {json_path.name}. Skipping.")
            p.disconnect(client_id)
            continue

        combined_mesh = o3d.geometry.TriangleMesh()
        for m in all_world_meshes:
            combined_mesh += m

        min_bound, max_bound = MIN_BOUND.copy(), MAX_BOUND.copy()
        if CONVERT_Z_UP_TO_Y_UP:
            R_convert = combined_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
            T_convert = np.eye(4)
            T_convert[:3, :3] = R_convert
            combined_mesh.transform(T_convert)

            for idx in link_data:
                old_T = link_data[idx]['T']
                new_T = T_convert @ old_T
                link_data[idx]['T'] = new_T
                link_data[idx]['pos'] = new_T[:3, 3]
                link_data[idx]['rotation'] = new_T[:3, :3]
                new_quat_xyzw = R.from_matrix(new_T[:3, :3]).as_quat()
                link_data[idx]['quat'] = np.array(new_quat_xyzw, dtype=np.float64)

            min_b_old, max_b_old = min_bound.copy(), max_bound.copy()
            min_bound[1], min_bound[2] = -max_b_old[2], min_b_old[1]
            max_bound[1], max_bound[2] = -min_b_old[2], max_b_old[1]

        if SAVE_INTERMEDIATE_MESH:
            debug_mesh_name = f"debug_{json_path.stem}.ply" # Use .stem for cleaner name
            out_mesh_path = Path(OUT_DIR) / debug_mesh_name
            o3d.io.write_triangle_mesh(str(out_mesh_path), combined_mesh, write_vertex_normals=False)
            # MODIFIED: Use tqdm.write
            tqdm.write(f"[debug] Saved intermediate mesh to: {out_mesh_path}")

        v = np.asarray(combined_mesh.vertices)
        v = np.clip(v, min_bound + EPS, max_bound - EPS)
        combined_mesh.vertices = o3d.utility.Vector3dVector(v)
        voxel_size = float(max_bound[0] - min_bound[0]) / float(VOXEL_RES)
        vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            combined_mesh, voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound
        )
        voxels = vg.get_voxels()
        if not voxels:
            pts_world = np.empty((0, 3), dtype=np.float64)
        else:
            ijk = np.array([vox.grid_index for vox in voxels], dtype=np.int32)
            pts_world = min_bound + (ijk.astype(np.float64) + 0.5) * voxel_size

        p.disconnect(client_id)
        pcd_tensor = torch.from_numpy(pts_world).float()

        fk_joints_dict = {}
        for idx, data in link_data.items():
            fk_joints_dict[idx] = {
                'T': torch.from_numpy(data['T']).float(),
                'pos': torch.from_numpy(data['pos']).float(),
                'rotation': torch.from_numpy(data['rotation']).float(),
                'quat': torch.from_numpy(data['quat']).float()
            }

        data_to_save = {
            "point_cloud": pcd_tensor,
            "joints": joint_states_to_save,
            "fk_joints": fk_joints_dict
        }

        output_filename = f"{json_path.name}".replace(".json", ".da")
        out_path = Path(OUT_DIR) / output_filename

        torch.save(data_to_save, out_path)
    print("\n[info] Batch processing complete.")


if __name__ == "__main__":
    main()
