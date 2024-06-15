import sys
import os

# torch must be imported before we import chamfer
import torch
import knn


def KNN(ref, query, k):
    assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
    with torch.no_grad():
        batch_size = ref.size(0)
        D, I = [], []
        for bi in range(batch_size):
            r, q = ref[bi].transpose(0, 1).contiguous(), query[bi].transpose(0, 1).contiguous()
            d, i = knn.knn(r.float(), q.float(), k)
            i -= 1
            d, i = d.transpose(0, 1).contiguous(), i.transpose(0, 1).contiguous()
            D.append(d)
            I.append(i)
        D = torch.stack(D, dim=0)
        I = torch.stack(I, dim=0)
    return D, I


bs = 1
n, m = 1000, 50
k = 10

ref = torch.rand(bs, 2, 3).cuda()
query = torch.rand(bs, m, 5).cuda()

ref = torch.tensor([[[0, 0, 0], [3, 3, 3], [10, 10, 10]]]).cuda()
query = torch.tensor([[[8, 8, 8], [1, 1, 1], [100, 100, 100]]]).cuda()

dist, indx = KNN(ref, query, 1)  # 32 x 50 x 10

a=0