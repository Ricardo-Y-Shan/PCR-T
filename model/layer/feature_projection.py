import torch
from torch import nn
import numpy as np

from utils.tools import camera_trans, camera_trans_inv, reduce_std
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, cameras, img_feats):
        """
            :param inputs: tensor of shape (bs, N, 3)
            :param camera: tensor of shape (bs, V, 5), where V = 3 <= number of cameras
            :param img_feats: list of 3 tensors with shapes: (bs*V, 16, 224, 224),  (bs*V, 64, 112, 112) and (bs*V, 256, 56, 56)
            :return tensor of shape (bs, N, K), where K = 1011 = 3 + 3 * 336
        """
        coord = inputs
        if cameras is not None:
            bs, V, _ = cameras.shape
            N = inputs.shape[1]

            point_origin = camera_trans_inv(cameras[:, 0, :], inputs)
            cameras = cameras.reshape([bs * V, 5])
            point_origin = point_origin.tile([1, V, 1]).reshape([bs * V, -1, 3])
            points = camera_trans(cameras, point_origin)
            # points: bs*V, N, 3
        else:  # pix3d
            bs, N, _ = inputs.shape
            V = 1
            points = inputs

        resolution = torch.tensor([224.0, 224.0])
        half_resolution = (resolution - 1) / 2
        w = (-248.0 * torch.divide(points[:, :, 0], points[:, :, 2]) + 0.5) / half_resolution[0]
        h = (248.0 * torch.divide(points[:, :, 1], points[:, :, 2]) + 0.5) / half_resolution[1]
        w = torch.nan_to_num(w)
        h = torch.nan_to_num(h)
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = []
        sample_points = torch.stack([w, h], dim=-1)
        for img_feat in img_feats:
            sample_result = F.grid_sample(img_feat, sample_points.unsqueeze(1))
            sample_result = torch.transpose(sample_result.squeeze(2), 1, 2)
            feats.append(sample_result)
        feats = torch.cat(feats, 2)  # (bs*V, N, 336)
        feats = feats.reshape([bs, V, N, -1])
        feats_max = torch.max(feats, dim=1)[0]
        feats_mean = torch.mean(feats, dim=1)
        feats_std = torch.std(feats, dim=1, unbiased=False)
        feats = torch.cat([coord, feats_max, feats_mean, feats_std], 2)

        return feats
