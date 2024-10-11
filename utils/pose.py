import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch.nn.functional as F


def vec2ss_matrix(vector, bs):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((bs,3,3), dtype=torch.float, device='cuda:0')
    ss_matrix[:, 0, 1] = -vector[:, 2]
    ss_matrix[:, 0, 2] = vector[:, 1]
    ss_matrix[:, 1, 0] = vector[:, 2]
    ss_matrix[:, 1, 2] = -vector[:, 0]
    ss_matrix[:, 2, 0] = -vector[:, 1]
    ss_matrix[:, 2, 1] = vector[:, 0]

    return ss_matrix

def z_rotation_matrix(angle, n, device):
    if n > 0:
        mat = torch.zeros((n, 3, 3), dtype=torch.float, device=device)
        coss = torch.cos(angle)
        sins = torch.sin(angle)
        mat[:, 0, 0] = coss
        mat[:, 0, 1] = -sins
        mat[:, 1, 0] = sins
        mat[:, 1, 1] = coss
        mat[:, 2, 2] = 1.
    else:
        mat = torch.zeros((3, 3), dtype=torch.float, device=device)
        coss = torch.cos(angle)
        sins = torch.sin(angle)
        mat[0, 0] = coss
        mat[0, 1] = -sins
        mat[1, 0] = sins
        mat[1, 1] = coss
        mat[2, 2] = 1.
    return mat

def activate_pts(pts, base_pts, thresh_dist=0.02):
    '''
    filter base_pts that are close enough from pts
    pts should be (n, n_probe, 3) format and base_pts should be  (n, n_near, 3)
    return a loss tensor that reflect (n,) in order 
    '''
    dist_mat = torch.cdist(pts, base_pts)
    act_pts = torch.exp(-(dist_mat) / thresh_dist)
    loss_pts = act_pts.flatten(1,-1).sum(dim=-1) / 2.
    return loss_pts


class Prob(nn.Module):
    def __init__(self, pts, cols, normals, k=0) -> None:
        '''
        convention: suppose gripper is (0, 0, -1) direction
        '''
        super(Prob, self).__init__()
        self.n, _ = pts.shape
        self.device = torch.device('cuda:0')
        self.pts = torch.from_numpy(pts).float().to(self.device)
        self.cols = torch.from_numpy(cols).float().to(self.device)
        self.normals = torch.from_numpy(normals).float().to(self.device)
        self.k = k
        self.geometric_center = torch.mean(self.pts, dim=0)

        if self.k > 0:
            dist_mat = torch.cdist(self.pts, self.pts, p=2)
            self.pair_dist, self.pair_index = torch.topk(dist_mat,dim=-1, k=k, largest=False, sorted=True) # per point min dist
            self.near_pts = self.gather_pts(self.pts, self.pair_index)
        else:
            self.near_pts = (self.pts * 1.)[None, ...].repeat(self.n, 1, 1)
            print(self.near_pts.shape)

        # self.long_dist = nn.Parameter(torch.normal(-0.15, 1e-6, size=(self.n, )))
        self.long_dist = torch.normal(-0.12, 1e-6, size=(self.n, ))
        self.roll_angle = nn.Parameter(torch.normal(0., 1e-6, size=(self.n, )))

        # intialization transformation
        trans = []
        for i in range(self.n):
            curr_normal = -normals[i][None, :]
            z_axis = np.array([0, 0, 1])[None, :]
            r, _ = R.align_vectors(curr_normal, z_axis)
            r_mat = np.array(r.as_matrix())
            # the r_mat describes the transformation from world 2 current point
            temp_trans = np.eye(4)
            temp_trans[:3, :3] = r_mat
            temp_trans[:3, 3] = pts[i]
            trans.append(temp_trans)
        self.initial_trans = torch.from_numpy(np.array(trans)).float().to(self.device)

    @torch.no_grad()
    def get_trans_mat_by_indice(self, ind):

        z_dist = self.long_dist[ind] * 1.
        roll_angle = self.roll_angle[ind] * 1.
        initial_trans = self.initial_trans[ind] * 1.

        trans = torch.zeros((4, 4), dtype=torch.float, device=self.device)
        trans[2, 3] = z_dist
        trans[:3, :3] = z_rotation_matrix(roll_angle, 0, self.device)
        trans[3, 3] = 1
        trans = initial_trans @ trans

        return trans

    def forward(self, gripper_pts):
        '''
        the inputs pts should all be in (n, n_probe,
        
          4)
        '''
        trans = torch.zeros((self.n, 4, 4), dtype=torch.float, device=self.device)
        trans[:, 2, 3] = self.long_dist
        trans[:, :3, :3] = z_rotation_matrix(self.roll_angle, self.n, self.device)
        trans[:, 3, 3] = 1
        # trans = torch.stack([m.inverse() for m in trans], dim=0)
        trans = torch.bmm(self.initial_trans, trans)

        probe_pts, collision_pts = gripper_pts
        probe_pts = (torch.bmm(trans, probe_pts.permute(0, 2, 1))).permute(0, 2, 1) # n, probe_n, 4
        collision_pts = (torch.bmm(trans, collision_pts.permute(0, 2, 1))).permute(0, 2, 1)

        probe_loss = activate_pts(probe_pts[..., :3], self.near_pts) # n, 
        collision_loss = activate_pts(collision_pts[..., :3], self.near_pts) # n,

        # geometric loss
        guide_vec = F.normalize((self.pts - self.geometric_center), p=2, dim=1)
        geo_loss = torch.abs(F.cosine_similarity(guide_vec, self.normals, dim=1))

        # print('probe: ', probe_loss[0].item(), 'coll: ', collision_loss[0].item(), 'geo: ', geo_loss[0].item())

        loss = - 0.001 * probe_loss + 0.01 * collision_loss


        return loss

    def gather_pts(self, pts, indices):
        '''
        pts: n, 3, indices: n, k
        return n, k, 3
        '''
        a = pts.unsqueeze(0).expand(self.n, -1, -1)
        b = indices.unsqueeze(-1).expand(-1, -1, 3)
        c = torch.gather(a, 1, b) 
        return c


