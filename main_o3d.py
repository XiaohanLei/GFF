import numpy as np
import pyrealsense2 as rs
import cv2
import math
from utils.sam import InteractiveSAM
from utils.camera import capture_image
import open3d as o3d
from utils.pointcloud import generate_pts, trans_pointcloud
from utils.pose import Prob
from utils.gripper_model import get_gripper_pts, get_gripper_meshes
from utils.vis_utils import ViserVisualizer, get_heatmap
import torch
from tqdm import tqdm
import time

def create_sphere_mesh(radius=0.05, resolution=20):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    return mesh

def create_cube_mesh(size=0.05):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    return mesh


def create_plus_mesh(size=0.05, thickness=0.01):
    vertices = np.array([
        [-size/2, -thickness/2, -thickness/2], [size/2, -thickness/2, -thickness/2],
        [-size/2, thickness/2, -thickness/2], [size/2, thickness/2, -thickness/2],
        [-size/2, -thickness/2, thickness/2], [size/2, -thickness/2, thickness/2],
        [-size/2, thickness/2, thickness/2], [size/2, thickness/2, thickness/2],
        
        [-thickness/2, -size/2, -thickness/2], [-thickness/2, size/2, -thickness/2],
        [thickness/2, -size/2, -thickness/2], [thickness/2, size/2, -thickness/2],
        [-thickness/2, -size/2, thickness/2], [-thickness/2, size/2, thickness/2],
        [thickness/2, -size/2, thickness/2], [thickness/2, size/2, thickness/2],
    ])
    
    faces = np.array([
        [0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,4,5], [0,5,1],
        [2,3,7], [2,7,6], [0,2,6], [0,6,4], [1,5,7], [1,7,3],
        [8,9,11], [8,11,10], [12,14,15], [12,15,13], [8,12,13], [8,13,9],
        [10,11,15], [10,15,14], [8,10,14], [8,14,12], [9,13,15], [9,15,11]
    ])
    
    return vertices, faces

def create_minus_mesh(size=0.05, thickness=0.01):
    vertices = np.array([
        [-size/2, -thickness/2, -thickness/2], [size/2, -thickness/2, -thickness/2],
        [-size/2, thickness/2, -thickness/2], [size/2, thickness/2, -thickness/2],
        [-size/2, -thickness/2, thickness/2], [size/2, -thickness/2, thickness/2],
        [-size/2, thickness/2, thickness/2], [size/2, thickness/2, thickness/2],
    ])
    
    faces = np.array([
        [0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,4,5], [0,5,1],
        [2,3,7], [2,7,6], [0,2,6], [0,6,4], [1,5,7], [1,7,3]
    ])
    
    return vertices, faces

def get_meshes(pts, sign, use_spehre=False):
    all_meshes = []
    if not use_spehre:
        plus_vertices, plus_faces = create_plus_mesh(size=0.007, thickness=0.001) # 016 0012
        minus_vertices, minus_faces = create_minus_mesh(size=0.007, thickness=0.001)
    charges = [sign for _ in range(len(pts))]
    for point, charge in zip(pts, charges):
        if charge == '+':
            if not use_spehre:
                vertices = plus_vertices + point
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(plus_faces)
                )
            else:
                mesh = create_sphere_mesh(radius=0.005)
                mesh.translate(point)
            mesh.paint_uniform_color([1, 0, 0])  # Red for positive
        else:
            if not use_spehre:
                vertices = minus_vertices + point
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(minus_faces)
                )
            else:
                mesh = create_cube_mesh(size=0.005)
                mesh.translate(point)
            mesh.paint_uniform_color([0, 0, 1])  # Blue for negative
        
        all_meshes.append(mesh)
    return all_meshes


def save_view(vis):
    ctr = vis.get_view_control()
    parameters = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("camera_params.json", parameters)
    print("视角参数已保存")
    return False

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

    # pts_vis = o3d.io.read_point_cloud('output/models/mug.ply')
    # pts_vis = o3d.io.read_point_cloud('models/000/nontextured.ply')
    # 1. Read the .npy files
    points = np.load('baymax_pts.npy')
    colors = np.load('baymax_cols.npy')

    # scene_name = '075'

    # obj_mesh = o3d.io.read_triangle_mesh(f'models/{scene_name}/textured.obj', enable_post_processing=True)
    # obj_mesh.compute_vertex_normals()
    # points = np.asarray(obj_mesh.vertices)



    # 2. Create an Open3D point cloud
    # pts_vis = o3d.io.read_point_cloud(f'models/{scene_name}/nontextured.ply')
    # o3d.visualization.draw_geometries([obj_mesh])

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(points[:, :3])
    pts_vis.colors = o3d.utility.Vector3dVector(colors / 255.)

    # downsample pointcloud
    pts_o = pts_vis
    pts_vis = pts_vis.voxel_down_sample(voxel_size=0.01)

    # # 可视化点云
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=800)
    vis.add_geometry(pts_o)

    # # 注册按键回调函数
    vis.register_key_callback(ord("S"), save_view)
    vis.run()
    vis.destroy_window()


    vis_down_size = 0.015 # 0.035

    extra_pts_vis = pts_vis.voxel_down_sample(voxel_size=vis_down_size)

    pts = np.asarray(pts_vis.points)
    cols = np.asarray(pts_vis.colors)

    pts_vis.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pts_vis.normals)

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()

    pts_vis_mesh = get_meshes(np.asarray(extra_pts_vis.points), "+")


    new_vis = o3d.visualization.Visualizer()
    new_vis.create_window(width=800, height=800)
    for mesh in pts_vis_mesh:
        new_vis.add_geometry(mesh)
        # new_vis.update_geometry(mesh)
    # 读取保存的视角参数
    loaded_parameters = o3d.io.read_pinhole_camera_parameters("camera_params.json")
    print(loaded_parameters)

    # 应用加载的视角参数
    ctr = new_vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(loaded_parameters, allow_arbitrary=True)

    new_vis.poll_events()
    new_vis.update_renderer()
    print("已加载保存的视角参数")
    new_vis.run()
    new_vis.destroy_window()

    # o3d.visualization.draw_geometries(pts_vis_mesh)


    # vis_ctrl = visualizer.get_view_control()
    # cam_params = vis_ctrl.convert_to_pinhole_camera_parameters()
    # print(cam_params.extrinsic)
    # trans = np.eye(4)
    # trans[1:3, :] *= 1
    # cam_params.extrinsic = trans
    # vis_ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    # visualizer.add_geometry(pts_vis_mesh)
    # visualizer.update_geometry(pts_vis_mesh)
    # visualizer.poll_events()
    # visualizer.update_renderer()

    # o3d.visualization.draw_geometries(pts_vis_mesh)

    gripper_pts = get_gripper_pts(pts.shape[0])

    attractive_pts_vis, repulsive_pts_vis = gripper_pts
    attractive_pts_vis = (attractive_pts_vis[0] * 1.).cpu().numpy()
    repulsive_pts_vis = (repulsive_pts_vis[0] * 1.).cpu().numpy()

    prob_gaussian = Prob(pts[:, :3], cols, normals).to('cuda:0')
    optimizer = torch.optim.Adam(params=prob_gaussian.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(prob_gaussian.parameters(), lr=0.0001)

    # time.sleep(10)

    for iter in tqdm(range(100)):
        loss = prob_gaussian(gripper_pts)
        loss_sum = loss.sum()

        optimizer.zero_grad()
        # loss.backward(torch.ones_like(loss))
        loss_sum.backward()
        optimizer.step()

        # visualize weighted gaussian
        heatmap = get_heatmap(loss, invert=True, cmap_name="jet")

        pts_vis.colors = o3d.utility.Vector3dVector(heatmap)
        # if iter % 10 == 0:
        #     o3d.visualization.draw_geometries([pts_vis])
        # visualizer.update_geometry(pts_vis)
        # visualizer.poll_events()
        # visualizer.update_renderer()

        with torch.no_grad():
            # if iter == 0:
            if True:
                min_indice = torch.argmin(loss)
            trans_mat = prob_gaussian.get_trans_mat_by_indice(min_indice)[None, ...]

            heatmap = torch.from_numpy(get_heatmap(torch.zeros((1,), device='cuda:0'),\
                                            invert=True, cmap_name="jet", exp=1)).to('cuda:0')


    # visualization gripper
    all_verts, all_faces = get_gripper_meshes(trans_mat)
    trans_at_pts = trans_pointcloud(attractive_pts_vis, trans_mat.cpu().numpy()[0])
    trans_rep_pts = trans_pointcloud(repulsive_pts_vis, trans_mat.cpu().numpy()[0])

    # vis_down_size = 0.02
    trans_at_pts_o3d = o3d.geometry.PointCloud()
    trans_at_pts_o3d.points = o3d.utility.Vector3dVector(trans_at_pts)
    # trans_at_pts_o3d = trans_at_pts_o3d.voxel_down_sample(voxel_size=vis_down_size)

    trans_rep_pts_o3d = o3d.geometry.PointCloud()
    trans_rep_pts_o3d.points = o3d.utility.Vector3dVector(trans_rep_pts)
    # trans_rep_pts_o3d = trans_rep_pts_o3d.voxel_down_sample(voxel_size=vis_down_size)

    at_mesh = get_meshes(np.asarray(trans_at_pts_o3d.points), '-')
    rep_mesh = get_meshes(np.asarray(trans_rep_pts_o3d.points), '+')

    o3d.visualization.draw_geometries(at_mesh + rep_mesh)


    # visualizer.add_geometry(at_mesh)
    # visualizer.update_geometry(at_mesh)
    # visualizer.poll_events()
    # visualizer.update_renderer()

    # visualizer.add_geometry(rep_mesh)
    # visualizer.update_geometry(rep_mesh)
    # visualizer.poll_events()
    # visualizer.update_renderer()

    
    for idx, (verts, faces) in enumerate(zip(all_verts, all_faces)):

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([obj_mesh, mesh])

    # o3d.visualization.draw_geometries([obj_mesh, mesh])

        # visualizer.add_geometry(mesh)
        # visualizer.update_geometry(mesh)
        # visualizer.poll_events()
        # visualizer.update_renderer()

            # time.sleep(1)
                
            




if __name__ == '__main__':
    main()