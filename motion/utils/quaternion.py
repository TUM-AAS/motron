import torch
import numpy as np


# PyTorch-backed implementations

def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qrot2(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    v_q = torch.zeros((v.shape[0], 4))
    v_q[..., 1:] = v

    return qmul(qmul(q, v_q), inverse(q))[..., 1:].view(original_shape)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q: torch.Tensor, order: str, epsilon: float = 0) -> torch.Tensor:
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def inverse(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4
    return q * torch.tensor([1, -1, -1, -1], device=q.device)


def qmat(q: torch.Tensor) -> torch.Tensor:
    q_shape = q.shape
    s = list(q_shape[:-1])
    quat = torch.nn.functional.normalize(q, dim=-1).contiguous()

    qw = quat[..., 0].contiguous().view(-1, 1)
    qx = quat[..., 1].contiguous().view(-1, 1)
    qy = quat[..., 2].contiguous().view(-1, 1)
    qz = quat[..., 3].contiguous().view(-1, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(s + [1, 3]), row1.view(s + [1, 3]), row2.view(s + [1, 3])), -2)  # batch*3*3

    return matrix


def qdist(q, r):
    return 2 * torch.acos((q*r).sum(dim=-1).abs().clamp(max=1.))


def matq(mat: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    if not torch.is_tensor(mat):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(mat)))

    repeat_mask = [1] * len(mat.shape)
    repeat_mask[-1] = 4

    rmat_t = torch.transpose(mat, -2, -1)

    mask_d2 = rmat_t[..., 2, 2] < epsilon

    mask_d0_d1 = rmat_t[..., 0, 0] > rmat_t[..., 1, 1]
    mask_d0_nd1 = rmat_t[..., 0, 0] < -rmat_t[..., 1, 1]

    t0 = 1 + rmat_t[..., 0, 0] - rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q0 = torch.stack([rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      t0, rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2]], -1)
    t0_rep = t0.unsqueeze(-1).unsqueeze(-1).repeat(*repeat_mask)

    t1 = 1 - rmat_t[..., 0, 0] + rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q1 = torch.stack([rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      t1, rmat_t[..., 1, 2] + rmat_t[..., 2, 1]], -1)
    t1_rep = t1.unsqueeze(-1).unsqueeze(-1).repeat(*repeat_mask)

    t2 = 1 - rmat_t[..., 0, 0] - rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q2 = torch.stack([rmat_t[..., 0, 1] - rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2],
                      rmat_t[..., 1, 2] + rmat_t[..., 2, 1], t2], -1)
    t2_rep = t2.unsqueeze(-1).unsqueeze(-1).repeat(*repeat_mask)

    t3 = 1 + rmat_t[..., 0, 0] + rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q3 = torch.stack([t3, rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] - rmat_t[..., 1, 0]], -1)
    t3_rep = t3.unsqueeze(-1).unsqueeze(-1).repeat(*repeat_mask)

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.unsqueeze(-1).type_as(q0)
    mask_c1 = mask_c1.unsqueeze(-1).type_as(q1)
    mask_c2 = mask_c2.unsqueeze(-1).type_as(q2)
    mask_c3 = mask_c3.unsqueeze(-1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0.unsqueeze(-1) + t1_rep * mask_c1.unsqueeze(-1) +  # noqa
                    t2_rep * mask_c2.unsqueeze(-1) + t3_rep * mask_c3.unsqueeze(-1)).squeeze(-2)  # noqa
    q *= 0.5
    return q


def ortho6dmat(ortho6d):
    x_raw = ortho6d[..., 0, :]
    y_raw = ortho6d[..., 1, :]

    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    x = x.unsqueeze(dim=-1)
    y = y.unsqueeze(dim=-1)
    z = z.unsqueeze(dim=-1)
    matrix = torch.cat((x, y, z), -1)
    return matrix


def eulerq(e: torch.Tensor, order: str) -> torch.Tensor:
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=-1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=-1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)

def quaternion_to_expmap(q: torch.tensor, epsilon=1e-8) -> torch.Tensor:
    """
    Convert quaternions to  axis-angle rotations (aka exponential maps).
    Formula from https://raw.githubusercontent.com/strasdat/Sophus/master/sophus/so3.hpp
    C. Hertzberg et al.:
    "Integrating Generic Sensor Fusion Algorithms with Sound State
    Representation through Encapsulation of Manifolds"
    Information Fusion, 2011
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.reshape(-1, 4)

    eps = torch.tensor([epsilon])

    w = q[..., 0:1]
    vec = q[..., 1:]
    n = torch.norm(vec, dim=-1)
    squared_n = torch.square(n)

    mask1 = (squared_n < torch.square(eps)).type_as(q)
    mask2 = (n < eps).type_as(q)
    mask3 = (w < 0).type_as(q)

    two_atan_nbyw_by_n = (mask1 * ((2. / w) - (2. / 3.) * squared_n * (w * torch.square(w)))
                          + (1 - mask1) * (
                              mask2 * ( mask3 * (np.pi / n) + (1 - mask3) * (- np.pi / n) )
                              + (1 - mask2) * (2 * torch.arctan(n / w) / n)
                          ))

    # if (squared_n < torch.square(eps)).all():
    #     squared_w = torch.square(w)
    #     two_atan_nbyw_by_n = (2. / w) - (2. / 3.) * squared_n * (w * squared_w)
    # else:
    #     if (torch.abs(n) < eps).all():
    #         if (w < 0).all():
    #             two_atan_nbyw_by_n = np.pi / n
    #         else:
    #             two_atan_nbyw_by_n = - np.pi / n
    #     else:
    #         two_atan_nbyw_by_n = 2. * torch.arctan(n / w) / n

    v = two_atan_nbyw_by_n * vec
    return v

# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qfix(q: np.array) -> np.array:
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def qpositive(q: torch.Tensor) -> torch.Tensor:
    mask = q[..., 0, 0] < 0
    return q * -mask.type_as(q).unsqueeze(dim=-1).unsqueeze(dim=-1) + q * (~mask).type_as(q).unsqueeze(dim=-1).unsqueeze(dim=-1)


def expmap_to_quaternion(e: np.array) -> np.array:
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e: np.array, order: str) -> np.array:
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)
