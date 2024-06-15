import torch
import torch.nn as nn

from model.layer.chamfer_wrapper import ChamferDist
from model.layer.emd_wrapper import emd


class PSGNLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()

        self.weights = options.loss

    def chamfer_loss(self, pc1, pc2):
        dist1, dist2, idx1, idx2 = self.chamfer_dist(pc1, pc2)
        loss = torch.mean(dist1) + self.weights.chamfer_opposite * torch.mean(dist2)
        return loss

    def emd_loss(self, pc1, pc2):
        cost = emd(pc1, pc2)
        loss = torch.mean(cost)
        return loss

    def forward(self, outputs, targets, epoch):
        view_num = int(targets["cameras"].shape[1])
        batch_size = int(targets["images"].shape[0])

        gt_coord = targets["points_resample"].float()  # bs*N*3
        gt_coord = gt_coord.unsqueeze(dim=1).expand(-1, view_num, -1, -1).reshape(batch_size * view_num, -1, 3)
        gt_mean = torch.mean(gt_coord, dim=1, keepdim=True)  # reduce average error
        gt_coord -= gt_mean

        points_all_views = outputs["points_all_views"]
        pc_mean = torch.mean(points_all_views, dim=1, keepdim=True)
        points_all_views -= pc_mean

        chamfer_loss = self.chamfer_loss(gt_coord, points_all_views)
        emd_loss = self.emd_loss(points_all_views, gt_coord)
        mean_loss = self.l1_loss(pc_mean, gt_mean)

        loss = 0
        loss += self.weights.chamfer * chamfer_loss
        loss += self.weights.emd * emd_loss
        loss += self.weights.mean * mean_loss

        return loss, {
            "chamfer": chamfer_loss,
            "emd": emd_loss,
            "mean": mean_loss,
        }
