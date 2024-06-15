import torch
import emd_cuda
import numpy as np


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def emd(target, query):
    """Earth Mover Distance (Approx)
        Args:
            target (torch.Tensor): (b, n, 3)
            query (torch.Tensor): (b, m, 3)
        Returns:
            cost (torch.Tensor): (b)
    """
    return EarthMoverDistanceFunction.apply(target, query)


batch_size = 10
n, m = 10000, 1024

xyz1 = torch.rand((batch_size, n, 3)).cuda()
xyz2 = torch.rand((batch_size, m, 3)).cuda()

cost=emd(xyz1,xyz2)
a=0