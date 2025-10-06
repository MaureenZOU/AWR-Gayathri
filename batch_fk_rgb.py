import os
import json
import math
import numpy as np
import pybullet as p
import open3d as o3d
from pathlib import Path
import torch
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ========= CONFIGURATION =========
# File paths
URDF_FILE = "/disk1/data/awr/robot/xhand_right/urdf/xhand_right.urdf"
JSON_DIR = "/disk1/data/awr/trellis/xhandr/joints_1/"
MESH_DIR = "/disk1/data/awr/robot/xhand_right/meshes"
OUT_DIR = "/disk1/data/awr/trellis/xhandr/joints_1_data_rgb"

# Voxelizer settings
VOXEL_RES = 512
MIN_BOUND = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
MAX_BOUND = np.array([0.5, 0.5, 0.5], dtype=np.float64)
EPS = 1e-6

# Rendering configuration
N_CAMERAS = 256
IMG_W = 512
IMG_H = 512
FOVY_DEG = 50.0  # vertical field-of-view for renderer
RGB_SUBDIR = "rgb"

# Camera sampling parameters (around Y-up after optional conversion)
ELEV_MIN_DEG = -60.0
ELEV_MAX_DEG = 60.0
RADIUS_SCALE_MIN = 0.6    # times bbox diagonal
RADIUS_SCALE_MAX = 1.2
SEED = 42  # for reproducibility, set to None to randomize

# Debug settings
SAVE_INTERMEDIATE_MESH = False
CONVERT_Z_UP_TO_Y_UP = True

# ========= UTILITY FUNCTIONS =========
@contextmanager
def suppress_c_stdout_stderr():
    """Context manager to suppress C-level stdout/stderr output."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

def _normalize(v):
    """Normalize a vector."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def T_from_pos_quat(pos, quat_xyzw):
    """Convert position and quaternion to 4x4 transformation matrix."""
    R_flat = p.getMatrixFromQuaternion(quat_xyzw)
    Rm = np.array(R_flat, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T

def get_mesh_scale_from_visual_tuple(vrec):
    """Extract mesh scale from PyBullet visual record."""
    if len(vrec) > 9 and isinstance(vrec[9], (list, tuple)) and len(vrec[9]) == 3:
        return np.array(vrec[9], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0], dtype=np.float64)

# ========= ROBOT LOADING AND PROCESSING =========
def load_robot_and_set_joints(urdf_path, json_path):
    """Load robot URDF and set joint states from JSON file."""
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

def gather_link_data(robot_id):
    """Gather position and orientation data for all robot links."""
    link_data = {}
    
    # Base link data
    base_pos, base_quat_xyzw = p.getBasePositionAndOrientation(robot_id)
    link_data[-1] = {
        'pos': np.array(base_pos, dtype=np.float64),
        'quat': np.array(base_quat_xyzw, dtype=np.float64),
        'rotation': np.array(p.getMatrixFromQuaternion(base_quat_xyzw), dtype=np.float64).reshape(3, 3),
        'T': T_from_pos_quat(base_pos, base_quat_xyzw)
    }
    
    # Joint link data
    num_joints = p.getNumJoints(robot_id)
    for link_idx in range(num_joints):
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

# ========= CAMERA AND RENDERING FUNCTIONS =========
def sample_camera_positions_around(center, bbox_diag, n, up_axis_y=True,
                                   elev_min_deg=-60.0, elev_max_deg=60.0,
                                   r_scale_min=0.6, r_scale_max=1.2, rng=None):
    """Sample camera positions around a central point on a sphere."""
    if rng is None:
        rng = np.random.default_rng()
    cams = []
    
    # Sample on a sphere in a Y-up world
    for _ in range(n):
        azim = rng.uniform(0.0, 2.0 * np.pi)
        elev = np.deg2rad(rng.uniform(elev_min_deg, elev_max_deg))
        
        # Direction in Y-up frame: (cos e * cos a, sin e, cos e * sin a)
        dir_world = np.array([
            math.cos(elev) * math.cos(azim),
            math.sin(elev),
            math.cos(elev) * math.sin(azim)
        ], dtype=np.float64)
        dir_world = _normalize(dir_world)
        
        radius = rng.uniform(r_scale_min * bbox_diag, r_scale_max * bbox_diag)
        cam_pos = center + radius * dir_world
        cams.append(cam_pos)
    return cams

def make_offscreen_renderer(width, height):
    """Create and configure an Open3D offscreen renderer."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # Configure lighting and background
    scene = renderer.scene
    scene.set_background([0.0, 0.0, 0.0, 1.0])  # black background
    
    # Add sun light
    scene.scene.set_sun_light(np.array([0.5, -1.0, 0.5]),  # direction
                              np.array([1.0, 1.0, 1.0]),   # color
                              75000.0)                     # intensity
    scene.scene.enable_sun_light(True)
    return renderer


def make_default_material():
    """Create default material for mesh rendering."""
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (0.7, 0.7, 0.7, 1.0)
    
    # Use the correct attribute names with 'base_' prefix
    mat.base_metallic = 0.0
    mat.base_roughness = 0.8
    mat.base_reflectance = 0.5
    
    return mat

def compute_camera_matrices(cam_center, lookat_center, world_up):
    """Compute camera transformation matrices using OpenCV convention."""
    # OpenCV camera frame convention in 3D: x right, y up, z forward
    forward = _normalize(lookat_center - cam_center)     # +Z_cam
    right = _normalize(np.cross(forward, world_up))      # +X_cam
    true_up = _normalize(np.cross(right, forward))       # +Y_cam
    
    # R_cw: camera->world (columns are camera axes in world coords)
    R_cw = np.column_stack((right, true_up, forward))
    # R_wc: world->camera
    R_wc = R_cw.T
    
    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = R_wc
    T_cam_world[:3, 3] = -R_wc @ cam_center
    
    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = R_cw
    T_world_cam[:3, 3] = cam_center
    
    return T_cam_world, T_world_cam

def render_rgb_views_for_mesh(combined_mesh, out_rgb_dir, n_cameras, fovy_deg,
                              img_w, img_h, up_is_y=True):
    """Render RGB images from multiple camera viewpoints around the mesh."""
    out_rgb_dir.mkdir(parents=True, exist_ok=True)

    # Build renderer and add mesh
    renderer = make_offscreen_renderer(img_w, img_h)
    scene = renderer.scene

    mat = make_default_material()
    geom_name = "robot_mesh"
    scene.add_geometry(geom_name, combined_mesh, mat)

    # Camera target = mesh AABB center in current coords
    aabb = combined_mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), dtype=np.float64)
    diag = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
    diag = float(diag) if diag > 0 else 1.0
    
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64) if up_is_y else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Sample camera positions
    rng = np.random.default_rng(SEED if SEED is not None else None)
    cam_centers = sample_camera_positions_around(
        center=center,
        bbox_diag=diag,
        n=n_cameras,
        up_axis_y=up_is_y,
        elev_min_deg=ELEV_MIN_DEG,
        elev_max_deg=ELEV_MAX_DEG,
        r_scale_min=RADIUS_SCALE_MIN,
        r_scale_max=RADIUS_SCALE_MAX,
        rng=rng
    )

    # Intrinsics from vertical FoV (square pixels => fx = fy)
    fovy_rad = math.radians(fovy_deg)
    fy = img_h / (2.0 * math.tan(0.5 * fovy_rad))
    fx = fy
    cx = img_w * 0.5
    cy = img_h * 0.5
    aspect = img_w / img_h

    rendered = []
    
    # Render each camera view
    for i, cam_center in enumerate(cam_centers):
        # Near/Far based on distance and object diagonal
        dist = float(np.linalg.norm(cam_center - center))
        near = max(0.01, dist - 1.5 * diag)
        far = dist + 1.5 * diag
        
        # Set camera for renderer (Filament)
        scene.camera.set_projection(fovy_deg, aspect, near, far,
                                    o3d.visualization.rendering.Camera.FovType.Vertical)
        scene.camera.look_at(center, cam_center, world_up)

        # Render image
        img_o3d = renderer.render_to_image()  # RGBA uint8
        np_img = np.asarray(img_o3d)
        rgb = np_img[..., :3].copy()  # HxWx3, uint8
        
        # Save image
        fname = f"rgb_{i:04d}.png"
        fpath = out_rgb_dir / fname
        o3d.io.write_image(str(fpath), o3d.geometry.Image(rgb), quality=9)

        # Compute camera matrices (OpenCV-style: world->camera)
        T_cam_world, T_world_cam = compute_camera_matrices(cam_center, center, world_up)

        cam_info = {
            "image_path": str(fpath),
            "T_cam_world": torch.from_numpy(T_cam_world).float(),
            "T_world_cam": torch.from_numpy(T_world_cam).float(),
            "intrinsics": {
                "fx": float(fx), "fy": float(fy),
                "cx": float(cx), "cy": float(cy),
                "width": int(img_w), "height": int(img_h),
                "fov_y_deg": float(fovy_deg),
                "near": float(near), "far": float(far),
            },
            "lookat_center": torch.from_numpy(center).float(),
            "cam_center": torch.from_numpy(cam_center).float(),
            "up": torch.from_numpy(world_up).float(),
        }
        rendered.append(cam_info)

    # Cleanup
    scene.remove_geometry(geom_name)
    del renderer
    return rendered

# ========= MAIN PROCESSING FUNCTION =========
def main():
    """Main processing function."""
    json_files = sorted(list(Path(JSON_DIR).rglob('joint_states_*.json')))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not json_files:
        print(f"[error] No 'joint_states_*.json' files found in {JSON_DIR}. Please check the path.")
        return

    print(f"[info] Found {len(json_files)} joint files to process.")

    for json_path in tqdm(json_files, desc="Processing joint files"):
        with open(json_path, "r") as f:
            joint_states_to_save = json.load(f)

        # Load robot and set joint states
        robot_id, client_id = load_robot_and_set_joints(URDF_FILE, json_path)
        link_data = gather_link_data(robot_id)

        # Process visual meshes
        vis_data = p.getVisualShapeData(robot_id)
        all_world_meshes = []

        for vrec in vis_data:
            link_idx, geom_type, filename_bytes = vrec[1], vrec[2], vrec[4]
            v_local_pos, v_local_quat = vrec[5], vrec[6]

            if geom_type != p.GEOM_MESH or not filename_bytes:
                continue

            # Load mesh file
            raw_filename = filename_bytes.decode('utf-8')
            base_filename = os.path.basename(raw_filename)
            resolved_filename = os.path.join(MESH_DIR, base_filename)

            if not os.path.exists(resolved_filename):
                tqdm.write(f"[warn] Skipping mesh, file not found: {resolved_filename}")
                continue

            mesh = o3d.io.read_triangle_mesh(resolved_filename)
            if mesh.is_empty():
                continue

            # Apply mesh transformations
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

        # Combine all meshes
        combined_mesh = o3d.geometry.TriangleMesh()
        for m in all_world_meshes:
            combined_mesh += m

        min_bound, max_bound = MIN_BOUND.copy(), MAX_BOUND.copy()

        # Optional: convert Z-up -> Y-up coordinate system
        if CONVERT_Z_UP_TO_Y_UP:
            R_convert = combined_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
            T_convert = np.eye(4)
            T_convert[:3, :3] = R_convert
            combined_mesh.transform(T_convert)
            
            # Update link data with new coordinate system
            for idx in link_data:
                old_T = link_data[idx]['T']
                new_T = T_convert @ old_T
                link_data[idx]['T'] = new_T
                link_data[idx]['pos'] = new_T[:3, 3]
                link_data[idx]['rotation'] = new_T[:3, :3]
                new_quat_xyzw = R.from_matrix(new_T[:3, :3]).as_quat()
                link_data[idx]['quat'] = np.array(new_quat_xyzw, dtype=np.float64)
            
            # Update bounds for new coordinate system
            min_b_old, max_b_old = min_bound.copy(), max_bound.copy()
            min_bound[1], min_bound[2] = -max_b_old[2], min_b_old[1]
            max_bound[1], max_bound[2] = -min_b_old[2], max_b_old[1]

        # Save intermediate mesh for debugging if requested
        if SAVE_INTERMEDIATE_MESH:
            debug_mesh_name = f"debug_{json_path.stem}.ply"
            out_mesh_path = Path(OUT_DIR) / debug_mesh_name
            o3d.io.write_triangle_mesh(str(out_mesh_path), combined_mesh, write_vertex_normals=False)
            tqdm.write(f"[debug] Saved intermediate mesh to: {out_mesh_path}")

        # Clip vertices to voxel bounds
        v = np.asarray(combined_mesh.vertices)
        v = np.clip(v, min_bound + EPS, max_bound - EPS)
        combined_mesh.vertices = o3d.utility.Vector3dVector(v)

        # Generate point cloud through voxelization
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

        # Render RGB views from multiple camera positions
        rgb_dir = os.path.join(Path(OUT_DIR), RGB_SUBDIR, json_path.stem)
        rgb_dir = Path(rgb_dir)
        rendered_cameras = render_rgb_views_for_mesh(
            combined_mesh=combined_mesh,
            out_rgb_dir=rgb_dir,
            n_cameras=N_CAMERAS,
            fovy_deg=FOVY_DEG,
            img_w=IMG_W,
            img_h=IMG_H,
            up_is_y=CONVERT_Z_UP_TO_Y_UP
        )
        p.disconnect(client_id)

        # Prepare data for saving
        pcd_tensor = torch.from_numpy(pts_world).float()

        fk_joints_dict = {}
        for idx, data in link_data.items():
            fk_joints_dict[idx] = {
                'T': torch.from_numpy(data['T']).float(),
                'pos': torch.from_numpy(data['pos']).float(),
                'rotation': torch.from_numpy(data['rotation']).float(),
                'quat': torch.from_numpy(data['quat']).float()
            }

        # Convert camera info to torch.save-friendly format
        cameras_to_save = []
        for cam in rendered_cameras:
            cameras_to_save.append({
                "image_path": cam["image_path"],
                "T_cam_world": cam["T_cam_world"],
                "T_world_cam": cam["T_world_cam"],
                "intrinsics": cam["intrinsics"],
                "lookat_center": cam["lookat_center"],
                "cam_center": cam["cam_center"],
                "up": cam["up"]
            })

        # Save all data
        data_to_save = {
            "point_cloud": pcd_tensor,
            "joints": joint_states_to_save,
            "fk_joints": fk_joints_dict,
            "cameras": cameras_to_save
        }

        output_filename = f"{json_path.name}".replace(".json", ".da")
        output_folder = os.path.join(OUT_DIR, "3d")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(output_folder, output_filename)
        torch.save(data_to_save, out_path)

    print("\n[info] Batch processing complete.")

if __name__ == "__main__":
    main()
