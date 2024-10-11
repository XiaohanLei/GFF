
import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_pts(color, depth, sam_mask, intrinsics):

    inv_int = np.linalg.inv(intrinsics)

    px, py = np.meshgrid(np.arange(640), np.arange(480))
    px = px.flatten()
    py = py.flatten()
    p = np.stack((px, py, np.ones_like(px)), axis=-1)

    depth_flatten = depth.flatten()
    mask = depth_flatten > 0
    sam_mask = sam_mask.flatten()
    depth_masked = depth_flatten[mask & sam_mask] / 1000.
    p_masked = p[mask & sam_mask]

    col = color.reshape(-1, 3)
    col = col[mask & sam_mask]


    pts_c = (inv_int @ p_masked.T).T * depth_masked[:, None]
    pts_c = np.concatenate((pts_c, np.ones((pts_c.shape[0], 1))), axis=-1)

    # crop mask
    crop_mask = pts_c[:, 2] < 3.0

    pts_c = pts_c[crop_mask]
    col_c = col[crop_mask]


    return pts_c, col_c

def trans_pointcloud(pts, trans):
    n, f = pts.shape
    if f == 3:
        extend = np.zeros(n, 1)
        pts = np.concatenate([pts, extend], axis=-1)
    trans_pts = (trans @ pts.transpose(1, 0)).transpose(1, 0)
    return trans_pts[:, :3]