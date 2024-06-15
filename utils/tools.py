import torch
import numpy as np


def print_number_of_params(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total: ", total_params, " trainable:", trainable_params)


def gather(target, index):
    '''
        target: [bs,N,d], d数据维度
        index: [bs,n,k], n条请求，每条请求k个点
        return: [bs,n,k,d]
    '''
    bs, N, d = target.shape
    _, n, k = index.shape

    index_base = torch.arange(bs, device=index.device).view(-1, 1, 1) * N
    index = index + index_base.expand(bs, n, k)
    index = index.reshape(bs * n * k)

    out = target.view(bs * N, d)[index, :]
    out = out.view(bs, n, k, d)

    return out


def recursive_detach(t):
    if isinstance(t, torch.Tensor):
        return t.detach()
    elif isinstance(t, list):
        return [recursive_detach(x) for x in t]
    elif isinstance(t, dict):
        return {k: recursive_detach(v) for k, v in t.items()}
    else:
        return t


# Cameras helper functions


def normal(v):
    norm = torch.norm(v, dim=1, keepdim=True)
    norm[norm == 0] = 1
    return torch.divide(v, norm.expand(-1, 3))


def cameraMat(cameras):
    '''
    cameras: bs*5

    return
    cm_mat: bs*3*3
    Z: bs*3
    '''

    theta = cameras[:, 0] * np.pi / 180.0
    camy = cameras[:, 3] * torch.sin(cameras[:, 1] * np.pi / 180.0)
    lens = cameras[:, 3] * torch.cos(cameras[:, 1] * np.pi / 180.0)
    camx = lens * torch.cos(theta)
    camz = lens * torch.sin(theta)
    Z = torch.stack([camx, camy, camz], dim=1)

    x = camy * torch.cos(theta + np.pi)
    z = camy * torch.sin(theta + np.pi)
    Y = torch.stack([x, lens, z], dim=1)
    X = torch.cross(Y, Z, dim=1)

    cm_mat = torch.stack([normal(X), normal(Y), normal(Z)], dim=1)
    return cm_mat, Z


def camera_trans(cameras, xyz):
    '''
        cameras: bs*5
        xyz: bs*N*3

        pt_trans: bs*N*3
    '''
    c, o = cameraMat(cameras)
    pt_trans = xyz - torch.unsqueeze(o, dim=1)
    pt_trans = torch.matmul(pt_trans, c.transpose(1, 2))

    return pt_trans


def camera_trans_inv(cameras, xyz):
    '''
    cameras: bs*5
    xyz: bs*N*3

    inv_xyz: bs*N*3
    '''
    c, o = cameraMat(cameras)
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    # inv_xyz = torch.linalg.solve(c, xyz.transpose(1, 2)).transpose(1, 2) + torch.unsqueeze(o, dim=1)
    inv_xyz = torch.matmul(xyz, torch.linalg.inv(c.transpose(1, 2))) + torch.unsqueeze(o, dim=1)
    # inv_xyz = []
    # for i in range(cameras.shape[0]):
    #    temp = torch.linalg.solve(c[i], xyz[i].transpose(0, 1)).transpose(0, 1) + o[i]
    #    inv_xyz.append(temp)
    # inv_xyz = torch.stack(inv_xyz, dim=0)
    return inv_xyz


def gather_nd_torch(params, indices, batch_dim=0):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.
    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.
    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.
    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].
    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])
    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])
    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = torch.mean(input=x, dim=axis, keepdim=True)
    devs_squared = torch.square(x - m)
    return torch.mean(input=devs_squared, dim=axis, keepdim=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims) + 1e-6)


def construct_feed_dict(pkl):
    stages = [pkl['stage1'], pkl['stage2'], pkl['stage3']]
    edges = []
    for i in range(1, 4):
        adj = pkl['stage{}'.format(i)][1]
        edges.append(adj[0])

    feed_dict = dict({'ellipsoid_feature_X': torch.tensor(pkl['coord'])})
    feed_dict.update({'edges': edges})
    feed_dict.update({'faces': pkl['faces']})
    feed_dict.update({'pool_idx': pkl['pool_idx']})  # 2 pool_idx
    feed_dict.update({'lape_idx': pkl['lape_idx']})
    feed_dict.update({'supports': stages})  # 3 supports
    feed_dict.update({'faces_triangle': pkl['faces_triangle']})  # 3 faces_traingles
    feed_dict.update({'sample_coord': pkl['sample_coord']})  # (43,3) sample_coord for deformation hypothesis
    feed_dict.update({'sample_adj': pkl['sample_cheb_dense']})

    return feed_dict
