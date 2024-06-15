import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import gather, print_number_of_params
from model.layer.knn_wrapper import knn_torch
from model.layer.feature_projection import FeatureProjection


class PositionFusion(nn.Module):
    def __init__(self, in_features, out_features, num_neighbours=10):
        super(PositionFusion, self).__init__()

        out_features = int(out_features / 2)
        self.num_neighbours = num_neighbours
        self.linear_coord = nn.Linear(6, out_features, bias=False)
        self.bn_coord = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.linear_feature = nn.Linear(in_features, out_features, bias=False)
        self.bn_feature = nn.BatchNorm1d(out_features)

    def knn_encoding(self, x, indices):
        x_knn = gather(x, indices)  # (bs, N, k, d_in)
        x_dup = x.unsqueeze(dim=2).repeat(1, 1, self.num_neighbours, 1)
        x_diff = x_knn - x_dup
        x_msg = torch.cat([x_dup, x_diff], dim=-1)  # (bs, N, k, 2*d_in)
        return x_msg

    def forward(self, xyz, feature):
        '''
        xyz: (bs, N, 3)
        feature: (bs, N, d), d = 1011
        '''

        _, indices = knn_torch(xyz, xyz, self.num_neighbours + 1)

        xyz_msg = self.knn_encoding(xyz, indices[:, :, 1:])  # (bs, N, k, 6)
        xyz_embedding = self.linear_coord(xyz_msg)  # (bs, N, k, d_out)
        xyz_embedding = self.relu(
            self.bn_coord(xyz_embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))  # (bs, N, k, d_out)
        xyz_embedding = torch.max(xyz_embedding, dim=2).values  # (bs, N, d_out)

        feature_embedding = self.linear_feature(feature)  # (bs, N, d_out)
        feature_embedding = self.relu(
            self.bn_feature(feature_embedding.transpose(1, 2)).transpose(2, 1))  # (bs, N, d_out)

        point_embedding = torch.cat([xyz_embedding, feature_embedding], dim=-1)  # (bs, N, 2*d_out)
        return point_embedding


class PositionHypothesisEncoding(nn.Module):
    def __init__(self, sample_coord=None):
        super(PositionHypothesisEncoding, self).__init__()

        position_hypothesis = torch.tensor([
            [0, 0, 0],
            [0.02, 0, 0],
            [0, 0.02, 0],
            [0, 0, 0.02],
            [-0.02, 0, 0],
            [0, -0.02, 0],
            [0, 0, -0.02],
        ]).float()
        if sample_coord is not None:
            position_hypothesis = torch.from_numpy(sample_coord).float()
        self.register_buffer('position_hypothesis', position_hypothesis)
        self.S = self.position_hypothesis.shape[0]

    def forward(self, xyz):
        '''
        xyz: (bs, N, 3)
        '''
        device = xyz.get_device()
        xyz_dup = xyz.unsqueeze(dim=2).tile([1, 1, self.S, 1])  # (bs, N, 7, 3)
        xyz_delta = self.position_hypothesis.unsqueeze(dim=0).unsqueeze(dim=0)  # (1, 1, 7, 3)
        xyz_hypothesis = xyz_dup + xyz_delta
        xyz_hypothesis = xyz_hypothesis.reshape([xyz.shape[0], -1, 3])  # (bs, N*7, 3)
        return xyz_hypothesis


class SelfAttention(nn.Module):
    def __init__(self, dim=256, heads=2, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
                                 nn.Dropout(dropout))

    def forward(self, inputs):
        attention_output, _ = self.self_attention(inputs, inputs, inputs)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
                                 nn.Dropout(dropout))

    def forward(self, query, key, value):
        attention_output, _ = self.cross_attention(query, key, value)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class Encoder(nn.Module):
    def __init__(self, feature_dim=256, layers=2, heads=2, dropout=0.1):
        super(Encoder, self).__init__()

        # attention_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=heads, dropout=dropout,
        #                                             dim_feedforward=feature_dim * 4, batch_first=True)
        # self.attn_encoder = nn.TransformerEncoder(attention_layer, layers)
        self.attn_encoder = SelfAttention(feature_dim, heads, dropout)

    def forward(self, feature):
        context = self.attn_encoder(feature)
        return context


class Decoder(nn.Module):
    def __init__(self, context_dim, heads=2, dropout=0.1):
        super(Decoder, self).__init__()

        self.context_dim = context_dim
        self.position_hypothesis = PositionHypothesisEncoding()
        # self.feature_projection = FeatureProjection()
        # self.feature_encoder = FeatureEncoding(in_dim=feat_dim, out_dim=context_dim)
        self.attn_decoder = CrossAttention(context_dim, heads, dropout)
        self.mlp_score = nn.Linear(context_dim, 1)
        layer = nn.TransformerEncoderLayer(d_model=self.context_dim, nhead=8, dropout=dropout,
                                           dim_feedforward=self.context_dim, batch_first=True)
        self.local_sa = nn.TransformerEncoder(layer, 3)
        # self.local_sa = SelfAttention(context_dim, heads, dropout)

    def forward(self, xyz, context, feature_hypo):
        '''
        xyz: (bs, N, 3)
        context: (bs, N, d)
        feature_hypo: (bs, N*S, d)
        '''
        bs, N, _ = xyz.shape
        # xyz_hypothesis = self.position_hypothesis(xyz)  # (bs, N*S, 3)
        # xyz_embedding = self.mlp_coord(xyz_hypothesis)  # (bs, N*S, d)
        xyz_embedding = self.attn_decoder(feature_hypo, context, context)  # (bs, N*S, d)
        xyz_embedding = xyz_embedding.reshape([-1, self.position_hypothesis.S, self.context_dim])  # (bs*N, S, d)
        xyz_embedding = self.local_sa(xyz_embedding)  # (bs*N, S, d)
        xyz_score = self.mlp_score(xyz_embedding)  # (bs*N, S, 1)
        xyz_score = torch.softmax(xyz_score, dim=1)  # (bs*N, S, 1)

        local_sample = self.position_hypothesis.position_hypothesis  # (S, 3)
        xyz_delta = xyz_score.tile([1, 1, 3]) * local_sample.tile([xyz_score.shape[0], 1, 1])  # (bs*N, S, 3)
        xyz_delta = xyz_delta.sum(dim=1)  # (bs*N, 3)

        xyz_next = xyz + xyz_delta.reshape([bs, N, 3])

        return xyz_next


class FeatureEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureEncoding, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.bn_coord = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)  # (bs, N, d_out)
        x = self.relu(self.bn_coord(x.transpose(1, 2)).transpose(1, 2))  # (bs, N, d_out)
        # x = self.linear2(x)
        return x


class PCRTransformer(nn.Module):
    def __init__(self, feat_dim, num_neighbours=4):
        super(PCRTransformer, self).__init__()

        context_dim = 128
        self.feature_projection = FeatureProjection()
        self.position_hypothesis = PositionHypothesisEncoding()
        # self.position_fusion = PositionFusion(in_features=feat_dim, out_features=context_dim,
        #                                      num_neighbours=num_neighbours)
        self.feature_encoder = FeatureEncoding(in_dim=feat_dim, out_dim=context_dim)
        self.encoder = Encoder(feature_dim=context_dim, heads=2)
        self.decoder = Decoder(context_dim=context_dim, heads=2)
        # self.decoder2 = Decoder(context_dim=context_dim, heads=2)

    def forward(self, points, cameras, img_feat):
        feature = self.feature_projection(points, cameras, img_feat)
        # point_embedding = self.position_fusion(points, feature)
        point_embedding = self.feature_encoder(feature)

        context = self.encoder(point_embedding)

        xyz_hypo = self.position_hypothesis(points)  # (bs, N*S, 3)
        feat_hypo = self.feature_projection(xyz_hypo, cameras, img_feat)  # (bs, N*S, 1011)
        feat_hypo = self.feature_encoder(feat_hypo)  # (bs, N*S, 256)

        points1 = self.decoder(points, context, feat_hypo)

        # xyz_hypo = self.position_hypothesis(points1)  # (bs, N*S, 3)
        # feat_hypo = self.feature_projection(xyz_hypo, cameras, img_feat)  # (bs, N*S, 1011)
        # feat_hypo = self.feature_encoder(feat_hypo)  # (bs, N*S, 256)
        #
        # points2 = self.decoder2(points1, context, feat_hypo)

        return [points1]


class LocalAttentionLayer(nn.Module):
    def __init__(self, feat_dim, sample_coord=None, layers=3, heads=8, dropout=0.1):
        super(LocalAttentionLayer, self).__init__()
        self.context_dim = 128
        self.position_hypothesis = PositionHypothesisEncoding(sample_coord)
        self.feature_projection = FeatureProjection()
        self.mlp1 = nn.Sequential(nn.Linear(feat_dim, self.context_dim), nn.ReLU(inplace=True))
        self.mlp2 = nn.Linear(self.context_dim, 1)
        attention_layer = nn.TransformerEncoderLayer(d_model=self.context_dim, nhead=heads, dropout=dropout,
                                                     dim_feedforward=self.context_dim, batch_first=True)
        self.attn_encoder = nn.TransformerEncoder(attention_layer, layers)

    def forward(self, xyz, cameras, img_feat):
        bs, N, _ = xyz.shape
        xyz_hypothesis = self.position_hypothesis(xyz)  # (bs, N*S, 3)
        feature = self.feature_projection(xyz_hypothesis, cameras, img_feat)  # (bs, N*S, d)
        context = self.mlp1(feature)  # (bs, N*S, 256)
        context = context.reshape([-1, self.position_hypothesis.S, self.context_dim])  # (bs*N, S, 256)
        context = self.attn_encoder(context)
        xyz_score = self.mlp2(context)  # (bs*N, S, 1)
        xyz_score = torch.softmax(xyz_score, dim=1)  # (bs*N, S, 1)

        local_sample = self.position_hypothesis.position_hypothesis  # (S, 3)
        xyz_delta = xyz_score.tile([1, 1, 3]) * local_sample.tile([xyz_score.shape[0], 1, 1])  # (bs*N, S, 3)
        xyz_delta = xyz_delta.sum(dim=1)  # (bs*N, 3)

        xyz_next = xyz + xyz_delta.reshape([bs, N, 3])
        return xyz_next


class LocalTransformer(nn.Module):
    def __init__(self, feat_dim, sample_coord=None, layers=2, heads=2, dropout=0.1):
        super(LocalTransformer, self).__init__()

        self.attn_layer1 = LocalAttentionLayer(feat_dim, sample_coord, layers, heads, dropout)
        self.attn_layer2 = LocalAttentionLayer(feat_dim, sample_coord, layers, heads, dropout)

    def forward(self, xyz, cameras, img_feat):
        xyz1 = self.attn_layer1(xyz, cameras, img_feat)
        xyz2 = self.attn_layer2(xyz1, cameras, img_feat)
        return [xyz1]


if __name__ == '__main__':
    xyz = torch.randn(2, 1024, 3)
    feature = torch.randn(2, 1024, 256)
    ps = PositionFusion(256, 128)
    context = ps(xyz, feature)
    a = 0
