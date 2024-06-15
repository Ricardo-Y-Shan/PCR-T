import torch


# import knn


# def knn_cuda(ref, query, k):
#     with torch.no_grad():
#         r, q = ref.transpose(0, 1).contiguous(), query.transpose(0, 1).contiguous()
#         d, i = knn.knn(r.float(), q.float(), k)
#         i -= 1
#         d, i = d.transpose(0, 1).contiguous(), i.transpose(0, 1).contiguous()
#     return d, i


def knn_torch(target, query, k):
    bs = target.shape[0]
    n = target.shape[1]
    m = query.shape[1]

    qq = (query ** 2).sum(dim=2, keepdim=True).expand(bs, m, n)
    tt = (target ** 2).sum(dim=2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = qq + tt - 2 * torch.matmul(query, target.transpose(1, 2))
    dist = torch.nan_to_num(dist, nan=torch.inf, neginf=torch.inf)
    values, indices = torch.topk(dist, k=k, dim=-1, largest=False)
    return values, indices


if __name__ == '__main__':
    from utils.tools import gather

    bs = 1
    N = 10
    target = torch.rand((bs, N, 3), requires_grad=True)
    query = torch.rand((bs, 2, 3))

    values, indices = knn_torch(target, query, 2)
    res = gather(target, indices)
    loss = torch.sum(res)
    loss.backward()
    print(target.grad)
    a = 0
