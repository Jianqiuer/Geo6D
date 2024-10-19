#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modified_pointnet import md_pointnet as spatial_encoder
from models.modified_resnet import md_resnet18 as resnet_extractor
from models.pointnet_util import knn_one_point

# from positional_encodings import PositionalEncodingPermute2D


def get_header(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, 640, kernel_size=1),
        nn.BatchNorm2d(640),
        nn.ReLU(inplace=True),
        nn.Conv2d(640, 256, kernel_size=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, out_channel, kernel_size=1)
    )


class XYZNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 strides=[2, 2, 1],
                 pn_conv_channels=[128, 128, 256, 512]):
        super(XYZNet, self).__init__()
        self.ft_1 = resnet_extractor(in_channel, strides)
        self.ft_2 = spatial_encoder(512, pn_conv_channels)

    def forward(self, xyzrgb):
        ft_1 = self.ft_1(xyzrgb)
        b, c, h, w = ft_1.size()
        rs_xyz = F.interpolate(xyzrgb[:, :3], (h, w), mode='bilinear')
        ft_2 = self.ft_2(ft_1, rs_xyz)
        ft_3 = torch.cat([ft_1, ft_2], dim=1)
        return ft_3, rs_xyz


class FCN6D(nn.Module):
    def __init__(self, num_class=21, in_channel=6):
        super(FCN6D, self).__init__()
        self.num_class = num_class

        # self.xyznet = XYZNet(8)
        self.xyznet = XYZNet(in_channel)

        self.trans = get_header(512 + 512 + 512, 3 * num_class)

        self.prim_x = get_header(512 + 512 + 512, 4 * num_class)

        self.score = get_header(512 + 512 + 512, num_class)

        if in_channel > 6:
            # geo6d: leanring object frame constraints
            self.geo_head = get_header(512 + 512 + 512, 3 * num_class)
        else:
            self.geo_head = None

    def forward(self, rgb, xyz, cls_ids, pe=False):
        xyzrgb = torch.cat([xyz, rgb], dim=1)
        # if pe:
        #     pos_enc_2d= PositionalEncodingPermute2D(xyzrgb.shape[1])
        #     global_pos_encoded = torch.zeros((1, *xyzrgb.shape[-3:]))
        #     position_encoding = pos_enc_2d(global_pos_encoded)
        #     xyzrgb += position_encoding.type_as(xyzrgb).to(xyzrgb.device)
        ft, rs_xyz = self.xyznet(xyzrgb)
        b, c, h, w = ft.size()

        px = self.prim_x(ft)
        tx = self.trans(ft)
        sc = F.sigmoid(self.score(ft))

        cls_ids = cls_ids.view(b).long()
        obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
        px = px.view(b, -1, 4, h, w)[obj_ids, cls_ids]
        tx = tx.view(b, -1, 3, h, w)[obj_ids, cls_ids]
        sc = sc.view(b, -1, h, w)[obj_ids, cls_ids]

        # pr[bs, 4, h, w], tx[bs, 3, h, w], xyz[bs, 3, h, w]
        tx = tx + rs_xyz
        geo_ft = None
        if self.geo_head is not None:
            geo_ft = self.geo_head(ft)
            geo_ft = geo_ft.view(b, -1, 3, h, w)[obj_ids, cls_ids].contiguous()
        del obj_ids

        # res is the final result
        return {'pred_r': px.contiguous(),
                'pred_t': tx.contiguous(),
                'pred_s': sc.contiguous(),
                'cls_id': cls_ids.contiguous(),
                'geo_ft': geo_ft}


class get_loss(nn.Module):
    def __init__(self, dataset, scoring_weight=0.01, geo_loss_weight=1.0):
        super(get_loss, self).__init__()
        self.prim_groups = dataset.prim_groups  # [obj_i:[gi:tensor[3, n]]]
        self.sym_list = dataset.get_sym_list()
        self.scoring_weight = scoring_weight
        self.geo_loss_weight = geo_loss_weight

    def quaternion_matrix(self, pr):
        R = torch.cat(((1.0 - 2.0 * (pr[2, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1),
                       (2.0 * pr[1, :] * pr[2, :] - 2.0 *
                        pr[0, :] * pr[3, :]).unsqueeze(dim=1),
                       (2.0 * pr[0, :] * pr[2, :] + 2.0 *
                        pr[1, :] * pr[3, :]).unsqueeze(dim=1),
                       (2.0 * pr[1, :] * pr[2, :] + 2.0 *
                           pr[3, :] * pr[0, :]).unsqueeze(dim=1),
                       (1.0 - 2.0 * (pr[1, :] ** 2 +
                                     pr[3, :] ** 2)).unsqueeze(dim=1),
                       (-2.0 * pr[0, :] * pr[1, :] + 2.0 *
                           pr[2, :] * pr[3, :]).unsqueeze(dim=1),
                       (-2.0 * pr[0, :] * pr[2, :] + 2.0 *
                           pr[1, :] * pr[3, :]).unsqueeze(dim=1),
                       (2.0 * pr[0, :] * pr[1, :] + 2.0 *
                           pr[2, :] * pr[3, :]).unsqueeze(dim=1),
                       (1.0 - 2.0 * (pr[1, :] ** 2 + pr[2, :] ** 2)).unsqueeze(dim=1)),
                      dim=1).contiguous().view(-1, 3, 3)  # [nv, 3, 3]
        return R

    def forward_geo(self, geo_ft, delta_abc_target, mask):
        h, w = geo_ft.size()[-2:]
        downsample_target = F.interpolate(
            delta_abc_target, (h, w), mode='bilinear')
        mask = mask.unsqueeze(dim=1)
        geo_ft = geo_ft * mask
        downsample_target = downsample_target * mask

        geo_loss = F.mse_loss(geo_ft, downsample_target, reduction='none')
        geo_loss = torch.sum(geo_loss * mask) / mask.sum()
        return geo_loss

    def forward(self, preds, mask, gt_r, gt_t, cls_ids, step=20):
        pred_r = preds['pred_r']
        pred_t = preds['pred_t']
        pred_score = preds['pred_s']

        bs, c, h, w = pred_r.size()
        pred_r = pred_r.view(bs, 4, h, w)
        pred_r = pred_r / torch.norm(pred_r, dim=1, keepdim=True)
        pred_r = pred_r.view(bs, 4, -1)
        pred_t = pred_t.view(bs, 3, -1)
        pred_score = pred_score.view(bs, -1)

        cls_ids = cls_ids.view(bs)

        # for one batch
        mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1)
        add_lst = torch.zeros(bs).cuda()
        loss_lst = torch.zeros(bs).cuda()
        for i in range(bs):
            # get mask id
            mk = mask[i].view(-1)
            valid_pixels = mk.nonzero().view(-1)
            num_valid = valid_pixels.size()[0]
            if num_valid < 1:
                continue
            if num_valid > 20:
                selected = [i * step for i in range(int(num_valid / step))]
                valid_pixels = valid_pixels[selected]
                num_valid = valid_pixels.size()[0]

            # get r, t, s, cls
            pr = pred_r[i][:, valid_pixels]  # [4, nv]
            pt = pred_t[i][:, valid_pixels]  # [3, nv]
            ps = pred_score[i][valid_pixels]  # [nv]

            # rotation matrix
            R_pre = self.quaternion_matrix(pr)
            R_tar = gt_r[i].view(1, 3, 3).repeat(
                num_valid, 1, 1).contiguous()  # [nv, 3, 3]
            t_tar = gt_t[i].view(1, 3).repeat(
                num_valid, 1).contiguous()  # [nv, 3]

            # group
            obj_grps = self.prim_groups[cls_ids[i]]
            add_ij = torch.zeros((len(obj_grps), num_valid)).cuda()
            for j, grp in enumerate(obj_grps):
                _, num_p = grp.size()
                grp = grp.cuda()
                grp = grp.view(1, 3, num_p).repeat(num_valid, 1, 1)

                npt = pt.permute(1, 0).contiguous().unsqueeze(
                    dim=2).repeat(1, 1, num_p)  # [nv, 3, np]
                ntt = t_tar.unsqueeze(dim=2).repeat(
                    1, 1, num_p).contiguous()  # [nv, 3, np]
                pred = torch.bmm(R_pre, grp) + npt  # [nv, 3, np]
                targ = torch.bmm(R_tar, grp) + ntt  # [nv, 3, np]

                pred = pred.unsqueeze(dim=1).repeat(
                    1, num_p, 1, 1).contiguous()  # [nv, np, 3, np]
                targ = targ.permute(0, 2, 1).unsqueeze(dim=3).repeat(
                    1, 1, 1, num_p).contiguous()  # [nv, np, 3, np]
                min_dist, _ = torch.min(torch.norm(
                    pred - targ, dim=2), dim=2)  # [nv, np]

                if len(obj_grps) == 3 and j == 2:
                    ########################
                    add_ij[j] = torch.max(min_dist, dim=1)[0]  # [nv]
                else:
                    add_ij[j] = torch.mean(min_dist, dim=1)  # [nv]

            # ADD(S)
            if len(obj_grps) == 3 and obj_grps[2].size()[1] > 1:
                add_i = torch.max(add_ij, dim=0)[0]  # [nv]
            else:
                add_i = torch.mean(add_ij, dim=0)  # [nv]
            add_lst[i] = torch.mean(add_i)
            loss_lst[i] = torch.mean(
                add_i * ps - self.scoring_weight * torch.log(ps))

        add = torch.mean(add_lst)
        loss = torch.mean(loss_lst)

        loss_dict = {'add_loss': loss.item(), 'add': add.item()}
        if preds["geo_ft"] is not None:
            geo_loss = self.forward_geo(
                preds["geo_ft"], preds["delta_abc_target"], mask)
            loss += self.geo_loss_weight * geo_loss
            loss_dict["geo_loss"] = geo_loss.item()
            loss_dict["loss"] = loss.item()
        return loss, loss_dict
