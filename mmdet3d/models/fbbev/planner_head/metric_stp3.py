'''
calculate planner metric same as stp3
'''
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from skimage.draw import polygon
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R

ego_width, ego_length = 1.85, 4.084

class PlanningMetric():
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.1]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.1]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        self.category_index = {
            'human':[2,3,4,5,6,7,8],
            'vehicle':[14,15,16,17,18,19,20,21,22,23]
        }
        
    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension



    def evaluate_single_coll(self, traj, segmentation, input_gt, gt_traj=None, index=None):
        '''
        traj: torch.Tensor (n_future, 2)
            自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
                
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        # 0.985793 is the distance betweem the LiDAR and the IMU(ego).

        import mmcv
        pts = np.array([
            [-self.H / 2. + 0.5 + 0.985793, self.W / 2.],
            [self.H / 2. + 0.5 + 0.985793, self.W / 2.],
            [self.H / 2. + 0.5 + 0.985793, -self.W / 2.],
            [-self.H / 2. + 0.5 + 0.985793, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        rc_ori = rc + (self.bx.cpu().numpy() / self.dx.cpu().numpy())


        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        rc_yaw = []
        rotate_angle = 0
        for i in range(traj.size(0)):
            delta = traj_with_ego[i+1] - traj_with_ego[i]
            cur_rotate_angle = torch.atan2(*delta[[1, 0]])
            if delta.norm()<1: cur_rotate_angle = 0
            rotate_angle = cur_rotate_angle
            rotate_angle = -torch.tensor(rotate_angle)
            rot_sin = torch.sin(rotate_angle)
            rot_cos = torch.cos(rotate_angle)
            rot_mat = torch.Tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
            tmp = rc_ori @ rot_mat.cpu().numpy() -  (self.bx.cpu().numpy() / self.dx.cpu().numpy())
            tmp = tmp.round().astype(np.int)
            rc_yaw.append(tmp)
           
        rc_yaw = np.stack(rc_yaw)

    
        # n_future, _ = traj.shape
        # trajs = traj.view(n_future, 1, 2)

        # trajs_ = copy.deepcopy(trajs)
        # trajs_ = trajs_ / self.dx.to(trajs.device)
        # trajs_ = trajs_.cpu().numpy() + rc # (n_future, 32, 2)

        # r = trajs_[:,:,0].astype(np.int32)
        # r = np.clip(r, 0, self.bev_dimension[0] - 1)

        # c = trajs_[:,:,1].astype(np.int32)
        # c = np.clip(c, 0, self.bev_dimension[1] - 1)

        # collision = np.full(n_future, False)
        # for t in range(n_future):
        #     rr = r[t]
        #     cc = c[t]
        #     I = np.logical_and(
        #         np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
        #         np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
        #     )
        #     collision[t] = np.any(segmentation[t,  cc[I], rr[I]].cpu().numpy())

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)

        trajs_ = copy.deepcopy(trajs)
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc_yaw # (n_future, 32, 2)

        r = trajs_[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision2 = np.full(n_future, False)
        # obs_occ = copy.deepcopy(segmentation).cpu().numpy() * 0
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
           
            collision2[t] = np.any(segmentation[t,  cc[I], rr[I]].cpu().numpy())
        return torch.from_numpy(collision2).to(device=traj.device)

    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation,
            index=None,
            ignore_gt=True,
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)

            xx, yy = trajs[i,:,0], trajs[i, :, 1]

            xi = ((-self.bx[0] + xx) / self.dx[0]).long()
            yi = ((-self.bx[1] + yy) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future).to(segmentation.device)
            # segmentation: B, T, H, W
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i],
                    segmentation[i],
                    gt_traj=gt_trajs[i],
                    input_gt=False,
                    index=index[i],
                    ).to(segmentation.device)
            if ignore_gt:
                obj_box_coll_sum += (gt_box_coll).long()                
            else:
                obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()
        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        
        return ade
