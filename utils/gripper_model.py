import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float
from typing import Tuple, Union

def get_gripper_pts(bs, z_offset=0.00) -> torch.Tensor:
    
    handle_pts_z = np.linspace(0., 0.1, 10)
    gripper_pts_z = np.linspace(0.1, 0.15, 5)
    horizon_pts_y = np.linspace(-0.045, 0.045, 9)
    # attractive_pts_z = np.linspace(0.14-z_offset, 0.16-z_offset, 4)
    z_min, z_max = 0.11, 0.16
    y_min, y_max = -0.035, 0.035
    interval = 0.005
    y_values = np.arange(y_min, y_max, interval)
    z_values = np.arange(z_min, z_max, interval)
    Y, Z = np.meshgrid(y_values, z_values)
    attractive_pts = np.column_stack([np.zeros_like(Y).flatten(), Y.flatten(), Z.flatten()])

    handle_pts = np.zeros((handle_pts_z.shape[0], 3))
    gripper_pts = np.zeros((2 * gripper_pts_z.shape[0], 3))
    horizon_pts = np.zeros((horizon_pts_y.shape[0], 3))
    # attractive_pts = np.zeros((attractive_pts_z.shape[0], 3))

    handle_pts[:, 2] = handle_pts_z

    horizon_pts[:, 1] = horizon_pts_y
    horizon_pts[:, 2] = 0.1

    gripper_pts[:(len(gripper_pts) // 2), 1] = 0.045
    gripper_pts[(len(gripper_pts) // 2):, 1] = -0.045
    gripper_pts[:(len(gripper_pts) // 2), 2] = gripper_pts_z
    gripper_pts[(len(gripper_pts) // 2):, 2] = gripper_pts_z

    # attractive_pts[:, 2] = attractive_pts_z

    handle_pts = torch.from_numpy(handle_pts).float().cuda()
    gripper_pts = torch.from_numpy(gripper_pts).float().cuda()
    horizon_pts = torch.from_numpy(horizon_pts).float().cuda()

    repulsive_pts = torch.cat((handle_pts, gripper_pts, horizon_pts), dim=0)
    repulsive_pts = torch.cat([repulsive_pts, torch.ones_like(repulsive_pts[..., :1])], dim=-1)
    repulsive_pts = repulsive_pts.repeat(bs, 1, 1)

    attractive_pts = torch.from_numpy(attractive_pts).float().cuda()
    attractive_pts = torch.cat([attractive_pts, torch.ones_like(attractive_pts[..., :1])], dim=-1)
    attractive_pts = attractive_pts.repeat(bs, 1, 1)

    return attractive_pts, repulsive_pts


def get_gripper_mesh(include_sphere: bool = True, radius: float = 0.003) -> o3d.geometry.TriangleMesh:
    """Get a skeleton gripper mesh."""
    # Create left and right fingers
    left_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.05)
    right_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.05)

    # Offset between bottom of fingers and the center of the grasp
    left_finger.translate((0, 0.045, 0.05 / 2 + 0.1))
    right_finger.translate((0, -0.045, 0.05 / 2 + 0.1))

    # Bar connecting the fingers, rotate so it's horizontal
    bar = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.09)
    bar.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    bar.translate((0, 0, 0.1))

    # Top extension of the gripper
    top_ext = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.1)
    top_ext.translate((0, 0, 0.05))

    if include_sphere:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        gripper_mesh = left_finger + right_finger + top_ext + bar + sphere
    else:
        gripper_mesh = left_finger + right_finger + top_ext + bar
    return gripper_mesh


def get_gripper_meshes(gripper_poses) -> Tuple[Float[np.ndarray, "n v 3"], Float[np.ndarray, "n f 3"]]:
    """
    Get vertices and faces for given gripper poses. Used for visualization purposes and returns the vertices and faces
    for each gripper pose.
    """
    # Get the gripper mesh and transform the vertices by each gripper pose
    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices = torch.from_numpy(vertices).float().to(gripper_poses.device)
    with torch.no_grad():
        extend_vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        all_vertices = (gripper_poses @ extend_vertices.T).permute(0, 2, 1)
        all_vertices = all_vertices[..., :3]

    # Get the faces
    faces = np.asarray(gripper_mesh.triangles)
    faces = torch.from_numpy(faces).to(gripper_poses.device)
    all_faces = faces.repeat(len(gripper_poses), 1, 1)

    # Convert to numpy
    all_vertices = all_vertices.cpu().numpy()
    all_faces = all_faces.cpu().numpy()
    return all_vertices, all_faces