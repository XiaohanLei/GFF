import numpy as np
import pyrealsense2 as rs
import cv2
import math
from utils.sam import InteractiveSAM
from utils.camera import capture_image
import open3d as o3d
from utils.pointcloud import generate_pts
from utils.pose import Prob
from utils.gripper_model import get_gripper_pts, get_gripper_meshes
from utils.vis_utils import ViserVisualizer, get_heatmap
import torch
from tqdm import tqdm
import time

def main():

    # rgb, depth, intrinsics = capture_image()
    # # intrinsics: height, width, fx, fy, ppx, ppy
    # int_mat = np.eye(3)
    # int_mat[0, 0], int_mat[0, 2], int_mat[1, 1], int_mat[1, 2] = intrinsics.fx, intrinsics.ppx, intrinsics.fy, intrinsics.ppy

    # sam = InteractiveSAM()
    # mask = sam.predict(rgb)
    # print('point cloud generated')

    # pts, cols = generate_pts(rgb, depth, mask, int_mat)

    # pts_vis = o3d.geometry.PointCloud()
    # pts_vis.points = o3d.utility.Vector3dVector(pts[:, :3])
    # pts_vis.colors = o3d.utility.Vector3dVector(cols / 255)

    # o3d.visualization.draw_geometries_with_editing([pts_vis])

    pts_vis = o3d.io.read_point_cloud('output/models/mug.ply')
    # pts_vis = o3d.io.read_point_cloud('models/000/nontextured.ply')

    # downsample pointcloud
    pts_vis = pts_vis.voxel_down_sample(voxel_size=0.002)

    pts = np.asarray(pts_vis.points)
    cols = np.asarray(pts_vis.colors)

    pts_vis.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pts_vis.normals)

    visualizer = ViserVisualizer('127.0.0.1', 15566)
    visualizer.add_point_cloud('pts', pts[:, :3], cols, point_size=0.002)

    gripper_pts = get_gripper_pts(pts.shape[0])

    prob_gaussian = Prob(pts[:, :3], cols, normals).to('cuda:0')
    optimizer = torch.optim.Adam(params=prob_gaussian.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(prob_gaussian.parameters(), lr=0.0001)

    # time.sleep(10)

    for iter in tqdm(range(10000)):
        loss = prob_gaussian(gripper_pts)
        loss_sum = loss.sum()

        optimizer.zero_grad()
        # loss.backward(torch.ones_like(loss))
        loss_sum.backward()
        optimizer.step()

        # visualize weighted gaussian
        heatmap = get_heatmap(loss, invert=True, cmap_name="jet")

        visualizer.add_point_cloud('pts', pts[:, :3], heatmap, point_size=0.002)

        with torch.no_grad():
            # if iter == 0:
            if True:
                min_indice = torch.argmin(loss)
            trans_mat = prob_gaussian.get_trans_mat_by_indice(min_indice)[None, ...]

            # visualization gripper
            all_verts, all_faces = get_gripper_meshes(trans_mat)

            heatmap = torch.from_numpy(get_heatmap(torch.zeros((1,), device='cuda:0'),\
                                                    invert=True, cmap_name="jet", exp=1)).to('cuda:0')
            for idx, (verts, faces) in enumerate(zip(all_verts, all_faces)):
                visualizer.add_mesh(f"grasps/grasp_{idx + 1}", verts, faces, heatmap[idx])

            # time.sleep(1)
                
            




if __name__ == '__main__':
    main()