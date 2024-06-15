import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer.chamfer_wrapper import ChamferDist
from model.layer.knn_wrapper import knn_torch
from model.layer.emd_wrapper import emd
from utils.tools import gather


class PCRLoss(nn.Module):
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

    def uniform_loss(self, pc):
        dist, _ = knn_torch(pc, pc, self.weights.uniform_control + 1)
        dist = dist[:, :, 1:]
        local_loss = torch.var(dist, dim=2, unbiased=False).mean()
        global_loss = torch.mean(dist, dim=2).var(dim=1, unbiased=False).mean()
        loss = self.weights.uniform_local * local_loss + self.weights.uniform_global * global_loss
        return loss

    def normal_loss(self, points_pred, points_gt, normal_gt):
        dist1, dist2, idx1, idx2 = self.chamfer_dist(points_pred, points_gt)
        nearest_normal = gather(normal_gt, idx1.unsqueeze(dim=-1))  # (bs, n, 1, 3)
        nearest_normal = F.normalize(nearest_normal, dim=-1)

        _, idx = knn_torch(points_pred, points_pred, self.weights.normal_control + 1)
        idx = idx[:, :, 1:]
        points_knn = gather(points_pred, idx)  # (bs, n, k, 3)
        vector_knn = points_knn - points_pred.unsqueeze(dim=2).repeat(1, 1, self.weights.normal_control, 1)
        vector_knn = F.normalize(vector_knn, dim=-1)  # (bs, n, k, 3)

        cosine = torch.einsum('...ij,...ij->...i', [nearest_normal, vector_knn])  # (bs, n, k)
        cosine = torch.abs(torch.sum(cosine, 2))
        return torch.mean(cosine)

    def forward(self, outputs, targets, epoch):
        # gt_coord = targets["points_resample"].float()
        gt_coord = targets["points"].float()
        gt_normal = targets["normals"].float()
        points = outputs["points"]

        chamfer_loss = 0.0
        emd_loss = 0.0
        uniform_loss = 0.0
        normal_loss = 0.0
        for p in points:
            chamfer_loss += self.chamfer_loss(gt_coord, p)
            emd_loss += self.emd_loss(p, gt_coord)
            # uniform_loss += self.uniform_loss(p)
            # normal_loss += self.normal_loss(p, gt_coord, gt_normal)

        # if 15 < epoch <= 20:
        #     uniform_loss *= 0.3
        # elif epoch > 20:
        #     uniform_loss *= 0.1

        loss = 0
        loss += self.weights.chamfer * chamfer_loss
        loss += self.weights.emd * emd_loss
        loss += self.weights.uniform * uniform_loss
        loss += self.weights.normal * normal_loss

        return loss, {
            "chamfer": chamfer_loss,
            "emd": emd_loss,
            # "uniform": uniform_loss,
            # "normal": normal_loss,
        }
