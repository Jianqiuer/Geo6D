import numpy as np
import open3d as o3d


def depth_to_xyz(depth, K, uv):
    u_i, v_i = uv[:, :, 0], uv[:, :, 1]
    f_x, c_x = K[0, 0], K[0, 2]
    f_y, c_y = K[1, 1], K[1, 2]
    x_i = (u_i - c_x) / f_x * depth
    y_i = (v_i - c_y) / f_y * depth
    return np.stack([x_i, y_i, depth], axis=-1)


def geo6d_reconstruct(xyz, mask, R=None, t=None):
    """
    reconstruct the input following the paper Geo6D
    xyz: [H, W, 3] 3D coordinates of the points in the camera frame calculated from depth
    mask: [H, W] valid mask
    R: [3, 3] rotation matrix (for training)
    t: [3] translation vector (for training)
    """

    # reference point generation
    depth_mask_xyz = xyz * mask[:, :, np.newaxis]
    choose = depth_mask_xyz[:, :, 2].flatten().nonzero()[0]
    mask_x = xyz[:, :, 0].flatten()[choose][:, np.newaxis]
    mask_y = xyz[:, :, 1].flatten()[choose][:, np.newaxis]
    mask_z = xyz[:, :, 2].flatten()[choose][:, np.newaxis]
    mask_xyz = np.concatenate((mask_x, mask_y, mask_z), axis=1)

    t_0 = mask_xyz.mean(axis=0).reshape(3)
    x_0, y_0, d_0 = t_0
    x_i, y_i, d_i = mask_xyz[:, 0], mask_xyz[:, 1], mask_xyz[:, 2]
    normalized_xyz = (mask_xyz - t_0.reshape((-1, 3)))

    # reconstruct input
    delta_u = x_i / d_i - x_0 / d_0
    delta_v = y_i / d_i - y_0 / d_0
    delta_d = d_i - d_0
    depth_d_d_0 = d_i * d_0
    t_0_depth = t_0.reshape(1, 3) / depth_d_d_0.reshape(-1, 1)

    input_data_valid = np.concatenate([delta_u.reshape(-1, 1), delta_v.reshape(-1, 1),
                                      delta_d.reshape(-1, 1), depth_d_d_0.reshape(-1, 1), t_0_depth], axis=-1)

    H, W = mask.shape
    input_data = np.zeros((H*W, input_data_valid.shape[-1]))
    input_data[choose, :] = input_data_valid
    input_data = input_data.reshape(H, W, -1)

    # reconstruct optimization targets
    if R is not None and t is not None:
        abc_i = np.matmul(mask_xyz - t.reshape(-1, 3), R)
        abc_0 = np.matmul(t_0.reshape(-1, 3) - t.reshape(-1, 3), R)
        delta_abc = abc_i / d_i.reshape(-1, 1) - abc_0 / d_0.reshape(1, 1)
        delta_t = t - t_0
        delta_abc_target = np.zeros((H*W, 3))
        delta_abc_target[choose, :] = delta_abc
        delta_abc_target = delta_abc_target.reshape(H, W, -1)
    else:
        delta_abc_target = delta_t = None

    target_data = {'R': R, 't': delta_t,
                   'centroid': t_0, 'delta_abc_target': delta_abc_target}

    return input_data, target_data
