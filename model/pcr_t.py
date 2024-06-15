import torch
from torch import nn

from model.psgn import PSGNModel
from model.transformer import PCRTransformer


class PCRModel(nn.Module):
    def __init__(self):
        super(PCRModel, self).__init__()

        self.psgn = PSGNModel(pretrained=True)
        feature_dim = 3 + 3 * (16 + 64 + 256)  # =1011
        self.transformer = PCRTransformer(feature_dim)

    def forward(self, imgs, cameras):
        # batch_size = imgs.shape[0]
        # view_num = cameras.shape[1] if cameras is not None else 1

        psgn_out = self.psgn(imgs, cameras)
        points_0, img_feat = psgn_out["points"], psgn_out["img_feat"]

        geo_feat = img_feat["geometry_feature"]
        points_out = self.transformer(points_0, cameras, geo_feat)
        return {
            "points": points_out,
            "points_before_gcn": points_0
        }
