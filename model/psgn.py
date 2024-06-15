import torch
from torch import nn

from global_config import PRETRAINED_WEIGHTS_PATH
from model.resnet import PCRResNet
from model.decoder_point import DecoderPoint

from utils.tools import camera_trans_inv, camera_trans


class PSGNModel(nn.Module):
    def __init__(self, pretrained=False):
        super(PSGNModel, self).__init__()

        self.encoder = PCRResNet()
        self.decoder_point = DecoderPoint()

        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS_PATH["psgn"], map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)

    def forward(self, imgs, cameras, view_num_out=None):
        if cameras is None:
            return self.forward_pix3d(imgs)

        batch_size = int(imgs.shape[0])
        view_num = int(cameras.shape[1])
        imgs = imgs.reshape(-1, 3, 224, 224)
        img_feat = self.encoder(imgs)
        latent = img_feat["latent_code"]
        points = self.decoder_point(latent)  # (bs*V, n, 3)

        points_origin = camera_trans_inv(cameras.reshape([-1, 5]), points)
        main_cameras = cameras[:, 0, :].tile([1, view_num]).reshape([-1, 5])
        points_current = camera_trans(main_cameras, points_origin)  # (bs*V, n, 3)

        if view_num_out is None:
            view_num_out = view_num
        l = torch.tensor(range(0, batch_size * view_num, view_num))
        # points = torch.cat([points_current[l + i] for i in range(view_num_out)], dim=1)

        points = points_current.reshape([batch_size, view_num, -1, 3])  # (bs, V, n, 3)
        view_mean = torch.mean(points, dim=2, keepdim=True)  # (bs, V, 1, 3)
        batch_mean = torch.mean(view_mean, dim=1, keepdim=True)  # (bs, 1, 1, 3)
        points_mean = points - view_mean + batch_mean
        points = points_mean.reshape([batch_size, -1, 3])

        return {
            "img_feat": img_feat,
            "points": points,
            "points_all_views": points_current,
        }

    def forward_pix3d(self, imgs):
        '''
        img: (bs, 3, 224, 224)
        '''
        img_feat = self.encoder(imgs)
        latent = img_feat["latent_code"]
        points = self.decoder_point(latent)  # (bs, n, 3)

        return {
            "img_feat": img_feat,
            "points": points,
        }
