import io
import json
import math
import os
import random
from curses import meta

import cv2
import h5py
import numpy as np
import numpy.ma as ma
import open3d as o3d
import scipy.io as scio
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from .es6d_data_processing import data_processing
from .geo6d_data_processing import geo6d_reconstruct


def read_file_list(file_path, sample_step=1):
    print('start read')
    file_list = []
    file_real = []
    file_syn = []
    input_file = open(file_path)
    step = 0
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        step += 1
        if step % sample_step != 0:
            continue
        if input_line[:5] == 'data/':
            file_real.append(input_line)
        else:
            file_syn.append(input_line)
        file_list.append(input_line)
    input_file.close()
    print('end read')
    return file_list, file_real, file_syn


def get_bbox(label):
    img_length = label.shape[1]
    img_width = label.shape[0]
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    wid = max(r_b, c_b)
    extend_wid = int(wid / 8)
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(wid / 2) - extend_wid
    rmax = center[0] + int(wid / 2) + extend_wid
    cmin = center[1] - int(wid / 2) - extend_wid
    cmax = center[1] + int(wid / 2) + extend_wid
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def remove_outliers(xyz, mask, visualize=False, max_dist=0.5):
    valid_mask = xyz[:, :, 2] * mask
    xyz_filter = xyz[:, :, 2] < 2.0
    valid_mask = np.logical_and(valid_mask > 0.0, xyz_filter)

    valid_xyz = xyz[valid_mask]
    valid_xyz_mean = valid_xyz.mean(axis=0)
    normalized_xyz = valid_xyz - valid_xyz_mean
    dist = np.linalg.norm(normalized_xyz, axis=1)
    valid_xyz = valid_xyz[dist < max_dist]

    valid_idx = [idx[dist < max_dist] for idx in np.where(valid_mask)]

    filtered_mask = np.zeros_like(mask)
    filtered_mask[tuple(valid_idx)] = 1.0

    if visualize:
        invalid_xyz = valid_xyz[np.where(dist > max_dist)]
        if len(invalid_xyz) > 0:
            diff_mask = np.logical_and(mask, 1 - filtered_mask)
            import cv2
            cv2.imwrite("diff_mask.png", diff_mask * 255)
            cv2.imwrite("mask.png", mask * 255)
            print(f"invalid_xyz: {invalid_xyz}")
            print(f"valid_xyz: {valid_xyz}")

    return filtered_mask


class PoseDataset(data.Dataset):
    def __init__(self, mode, opt, add_noise=False):
        self.mode = mode
        if mode == 'train':
            train_list_name = getattr(opt, 'train_list_name', 'train_list.txt')
            self.list, self.real, self.syn = read_file_list(
                os.path.join(opt.dataset_root, train_list_name))
        elif mode == 'test':
            self.list, self.real, self.syn = read_file_list(
                os.path.join(opt.dataset_root, 'val_list.txt'))
        num_pt = opt.num_points
        root = opt.dataset_root
        self.num_pt = num_pt
        self.root = root
        self.opt = opt
        self.add_noise = add_noise
        self.aug_mode = getattr(opt, 'aug_mode', 'mask')
        class_file = open('datasets/ycb/dataset_config/classes.txt')
        self.cld = []
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            input_cloud = o3d.io.read_point_cloud(
                '{0}/ycb_models/{1}/textured.pcd'.format("datasets/ycb/", class_input[:-1]))
            raw_xyz = torch.tensor(np.asarray(input_cloud.points).reshape(
                (1, -1, 3)), dtype=torch.float32)
            xyz_ids = farthest_point_sample(raw_xyz, num_pt).cpu().numpy()
            raw_xyz = np.asarray(input_cloud.points).astype(np.float32)
            self.cld.append(raw_xyz[xyz_ids[0, :], :])

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        self.prim_groups = []

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.resize_img_width = 128
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(
            mean=[0.485*255.0, 0.456*255.0, 0.406*255.0], std=[0.229*255.0, 0.224*255.0, 0.225*255.0])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]  # start with 0
        # self.obj_radius = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
        #                    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        self.obj_radius = [1.0] * 21

        self.front_num = 2
        self.aug_mask = getattr(opt, 'augmask', False)
        self.vis = getattr(opt, 'vis', False)
        self.pe = getattr(opt, 'pe', False)
        self.mask_xyz_only = getattr(opt, 'mask_xyz_only', False)
        self.geo6d_mode = getattr(opt, 'geo6d_mode', False)

        # load primitives
        with open('datasets/ycb/dataset_config/tsn_meta.json', 'r') as f:
            prim_groups = json.load(f)
            for i, prim in enumerate(prim_groups):
                tmp = []
                for grp in prim['groups']:
                    tmp.append(torch.tensor(grp, dtype=torch.float).permute(
                        1, 0).contiguous() / self.obj_radius[i])
                self.prim_groups.append(tmp)

        print(f'total test number: {len(self.list)}')
        self.ymap = np.array([[j for i in range(640)]
                             for j in range(480)]).astype(np.float32)
        self.xmap = np.array([[i for i in range(640)]
                             for j in range(480)]).astype(np.float32)

    def phase_item(self, index, sample_list, filter_size=50, resize_img_width=None, resize_img_height=None, pe=None):
        valid = False
        while not valid:
            file_name = sample_list[index][:-
                                           2] if sample_list[index][-2] == '_' else sample_list[index][:-3]
            ins_idx = sample_list[index].split('_')[-1]
            meta = scio.loadmat(
                '{0}/{1}-meta.mat'.format(self.root, file_name))
            cam_scale = meta['factor_depth'][0][0]
            img = Image.open('{0}/{1}-color.png'.format(self.root, file_name))
            img = np.array(img)

            depth = Image.open(
                '{0}/{1}-depth.png'.format(self.root, file_name))
            depth = np.array(depth).astype(np.float32)
            depth /= cam_scale
            label = np.array(Image.open(
                '{0}/{1}-label.png'.format(self.root, file_name)))
            if file_name[:8] != 'data_syn' and int(file_name[5:9]) >= 60:
                cam_cx = 323.7872
                cam_cy = 279.6921
                cam_fx = 1077.836
                cam_fy = 1078.189
            else:
                cam_cx = 312.9869
                cam_cy = 241.3109
                cam_fx = 1066.778
                cam_fy = 1067.487
            camera_info = {'cam_cx': cam_cx, 'cam_cy': cam_cy,
                           'cam_fx': cam_fx, 'cam_fy': cam_fy}
            obj = meta['cls_indexes'].flatten().astype(np.int32)
            ins_idx = int(ins_idx)
            mask_label = ma.getmaskarray(ma.masked_equal(
                label, obj[ins_idx])).astype(np.float32)

            mask = mask_label

            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            if cmax - cmin > 0 and rmax - rmin > 0:
                if img.size >= 3 * 640 * 480:
                    img_crop = img[:, :, :3][rmin:rmax, cmin:cmax, :]
                else:
                    print(f'file_name: {file_name}')
                    print(f"img.shape: {img.shape}")
                    print(f"rmin:rmax, cmin:cmax: {(rmin, rmax, cmin, cmax)}")
                    continue
                img_mask_crop = mask_label[rmin:rmax, cmin:cmax]
                depth_crop = depth[rmin:rmax, cmin:cmax,
                                   np.newaxis].astype(np.float32)

                # choose num_pt from the selected object or pad to num_pt
                xmap_masked = self.xmap[rmin:rmax, cmin:cmax, np.newaxis]
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax, np.newaxis]

                pt2 = depth_crop
                pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
                depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2)
                rgb = img_crop.astype(np.float32)
                mask = img_mask_crop.astype(np.float32)
                xyz = depth_xyz.astype(np.float32)
                valid = True
                break
            else:
                index = np.random.choice(len(sample_list))
        pose = meta['poses'][:, :, ins_idx].astype(np.float32)
        cls_id = obj[ins_idx].astype(np.int64)
        position_encoding = None
        if resize_img_width is not None and resize_img_height is not None:
            rgb, xyz, mask, position_encoding = resize(
                rgb, xyz, mask, resize_img_width, resize_img_height, pe=position_encoding)
        else:
            rgb, xyz, mask, position_encoding = resize(
                rgb, xyz, mask, self.resize_img_width, self.resize_img_width, pe=position_encoding)

        return rgb, xyz, mask, pose, cls_id, camera_info, position_encoding

    def online_aug(self, rgb, xyz, mask, real=True, addfront_p=0.75, blur_p=0.75):
        h, w, c = rgb.shape
        min_numpoint = h * w / 20.0
        if (random.uniform(0.0, 1.0) < addfront_p):
            if (real):
                sample_list = self.syn
            else:
                sample_list = self.real
            seed = np.random.choice(len(sample_list))
            syn_rgb, syn_xyz, syn_mask, _, _, _, _ = self.phase_item(
                seed, sample_list, 0, w, h)

            if (syn_mask.sum() > min_numpoint):
                for i in range(5):
                    # rotate and translate syn_obj
                    syn_rgb1, syn_xyz1, syn_mask1 = random_rotation_translation(
                        syn_rgb, syn_xyz, syn_mask, 90, [0.4, 0.8])
                    # paste synthesized object in front of real object
                    new_rgb, new_xyz, new_mask = paste_two_objects(
                        syn_rgb1, syn_xyz1, syn_mask1, rgb, xyz, mask)
                    if (new_mask.sum() / mask.sum() > 0.3 and new_mask.sum() > w * h / 20):
                        if (xyz * new_mask[:, :, np.newaxis])[:, :, 2].flatten().nonzero()[0].size < 50:
                            # print('error mask')
                            continue
                        mask = new_mask
                        if getattr(self.opt, 'aug_rgb', False):
                            rgb = new_rgb
                        if getattr(self.opt, 'aug_xyz', False):
                            xyz = new_xyz
                        break

        if (real == False) and getattr(self.opt, 'aug_rgb_sim', False):
            sample_list = self.real
            seed = np.random.choice(len(sample_list))
            syn_rgb, syn_xyz, syn_mask, _, _, _, _ = self.phase_item(
                seed, sample_list, 0, w, h)
            # z_offset = -0.05 - float(random.uniform(0.0, 0.1))
            # syn_xyz[:, :, 2] = syn_xyz[:, :, 2] + z_offset
            back_mask = (rgb[:, :, 0] == 0).astype(np.float32)
            front_mask = (rgb[:, :, 0] > 0).astype(np.float32)
            rgb, new_xyz, _ = paste_two_objects(
                rgb, xyz, front_mask, syn_rgb, syn_xyz, back_mask)
            if getattr(self.opt, 'aug_xyz_sim', False):
                xyz = new_xyz
            # randomly blur
            if (random.uniform(0.0, 1.0) < blur_p):
                rgb = Image.fromarray(rgb.astype('uint8')).convert('RGB')
                rgb = rgb.filter(ImageFilter.BoxBlur(random.choice([1])))
                rgb = np.asarray(rgb).astype(np.float32)

        if self.aug_mode == 'mask':
            return mask
        elif self.aug_mode == 'mask_rgb':
            return rgb, mask

    def __getitem__(self, index):
        rgb, xyz, mask, pose, cls_id, camera_info, pe = self.phase_item(
            index=index, sample_list=self.list, pe=self.pe)
        # augmentation
        if (self.mode == 'train'):
            if (self.list[index][:8] != 'data_syn'):
                # print("gggggggg")
                if self.aug_mode == 'mask':
                    mask = self.online_aug(rgb, xyz, mask, True)
                elif self.aug_mode == 'mask_rgb':
                    rgb, mask = self.online_aug(rgb, xyz, mask, True)
            else:
                if self.aug_mode == 'mask':
                    mask = self.online_aug(rgb, xyz, mask, False)
                elif self.aug_mode == 'mask_rgb':
                    rgb, mask = self.online_aug(rgb, xyz, mask, False)
            rgb = np.asarray(self.trancolor(Image.fromarray(
                rgb.astype('uint8')))).astype(np.float32)
        if self.vis:
            os.makedirs('vis_dir/', exist_ok=True)
            filename = '_'.join(self.list[index].split('/'))
            cv2.imwrite(f'vis_dir/{filename}_rgb.png',
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'vis_dir/{filename}_mask.png', mask * 255)

        mask = remove_outliers(xyz, mask, visualize=False)

        if self.geo6d_mode:
            input_data, target_data = geo6d_reconstruct(
                xyz, mask, pose[:, 0:3], pose[:, 3])
        else:
            input_data, target_data = data_processing(
                xyz, mask, pose[:, 0:3], pose[:, 3])

        model_points = self.cld[cls_id - 1].T

        # normalize with obj radius
        radius = self.obj_radius[cls_id - 1]
        input_data[:, :, 0:3] = input_data[:, :, 0:3] / radius
        target_data['t'] = target_data['t'] / \
            radius  # TODO: donot normalize target_t
        model_points = model_points / radius

        if self.aug_mask:
            if not self.mask_xyz_only:
                rgb = rgb * mask[:, :, np.newaxis]
            input_data[:, :, 0:3] = input_data[:,
                                               :, 0:3] * mask[:, :, np.newaxis]

        if (mask.sum() == 0.0):
            mask = np.ones(mask.shape, dtype=np.float32)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0)
        rgb = torch.from_numpy(rgb.astype(np.float32)
                               ).permute(2, 0, 1).contiguous()
        rgb = self.norm(rgb)
        input_data = torch.from_numpy(input_data.astype(np.float32)).permute(
            2, 0, 1).contiguous()
        if pe is not None and self.pe:
            pe = torch.from_numpy(pe.astype(np.float32)).permute(
                2, 0, 1).contiguous()
            pe = mask * pe
            rgb = rgb + pe[:3, :, :]
            input_data = input_data + pe[3:, :, :]

        # process target data
        centroid = torch.from_numpy(
            target_data['centroid'].astype(np.float32)).view(3)
        target_r = target_data['R']
        target_t = target_data['t']

        if self.geo6d_mode:
            delta_abc_target = target_data['delta_abc_target']  # / radius
            delta_abc_target = torch.from_numpy(
                delta_abc_target.astype(np.float32)).permute(2, 0, 1).contiguous()
        else:
            delta_abc_target = torch.zeros_like(input_data)

        return {
            'rgb': rgb,
            'xyz': input_data,
            'delta_abc_target': delta_abc_target,
            'mask': mask,
            'target_r': torch.from_numpy(target_r.astype(np.float32)).view(3, 3),
            'target_t': torch.from_numpy(target_t.astype(np.float32)).view(3),
            'model_xyz': torch.from_numpy(model_points.astype(np.float32)),
            'class_id': torch.LongTensor([int(cls_id)-1]),
            'mean_xyz': centroid}  # 0,1...

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def resize(rgb, xyz, mask, width, height, pe=None):
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(
        dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(
        dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)
                            ).unsqueeze(dim=0).unsqueeze(dim=0)

    rgb = F.interpolate(rgb, size=(height, width), mode='bilinear').squeeze(
        dim=0).permute(1, 2, 0).contiguous()
    xyz = F.interpolate(xyz, size=(height, width), mode='nearest').squeeze(
        dim=0).permute(1, 2, 0).contiguous()
    mask = F.interpolate(mask, size=(height, width),
                         mode='nearest').squeeze(dim=0).squeeze(dim=0)
    if pe is not None:
        pe = F.interpolate(pe, size=(height, width), mode='nearest').squeeze(
            dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy(), pe


def random_rotation_translation(rgb, xyz, mask, degree_range, trans_range):
    h, w, c = rgb.shape
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(
        dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(
        dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)
                            ).unsqueeze(dim=0).unsqueeze(dim=0)

    angle = float(random.uniform(-degree_range, degree_range)) * \
        math.pi / 180.0
    trans1 = random.choice([float(random.uniform(
        trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])
    trans2 = random.choice([float(random.uniform(
        trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])

    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), trans1],
        [math.sin(angle), math.cos(angle), trans2]
    ], dtype=torch.float)

    grid = F.affine_grid(theta.unsqueeze(0), rgb.size())
    rgb = F.grid_sample(rgb, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.grid_sample(xyz, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.grid_sample(mask, grid, mode='nearest').squeeze(
        dim=0).squeeze(dim=0)

    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy()


def paste_two_objects(f_rgb, f_xyz, f_mask, b_rgb, b_xyz, b_mask):
    mask = b_mask - b_mask * f_mask
    # both_mask = mask + f_mask
    rgb = b_rgb * (1 - f_mask[:, :, np.newaxis]) + \
        f_rgb * f_mask[:, :, np.newaxis]
    xyz = b_xyz * (1 - f_mask[:, :, np.newaxis]) + \
        f_xyz * f_mask[:, :, np.newaxis]
    return rgb, xyz, mask


if __name__ == '__main__':
    pass
