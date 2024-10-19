import numpy as np


def data_processing(xyz, mask, R, t):
    depth_xyz = xyz
    depth_mask_xyz = depth_xyz * mask[:, :, np.newaxis]
    choose = depth_mask_xyz[:, :, 2].flatten().nonzero()[0]

    # 3D coordinates normalization
    mask_x = depth_xyz[:, :, 0].flatten()[choose][:, np.newaxis]
    mask_y = depth_xyz[:, :, 1].flatten()[choose][:, np.newaxis]
    mask_z = depth_xyz[:, :, 2].flatten()[choose][:, np.newaxis]
    mask_xyz = np.concatenate((mask_x, mask_y, mask_z), axis=1)
    mean_xyz = mask_xyz.mean(axis=0).reshape((1, 1, 3))
    normalized_xyz = (xyz - mean_xyz)
    input_data = normalized_xyz * mask[:, :, np.newaxis]

    # target data
    delta_t = t - mean_xyz.reshape(3)
    target_data = {'R': R, 't': delta_t, 'centroid': mean_xyz}

    return input_data, target_data
