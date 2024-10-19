import argparse
import math
import os
import random
import time
import warnings
# import lib
from builtins import ValueError
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from datasets.ycb.dataset_prim_tsn_geo6d import \
    PoseDataset as PoseDataset_ycb_online
from lib.utils import post_processing_ycb_quaternion_wi_vote as post_processing
from lib.utils import save_pred_and_gt_json, setup_logger, warnup_lr
from lib.ycb_evaluator import YCBEval
from models import XYZNet_4_GADD_5 as icp_pose_xyz
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0, help='gpu number')
parser.add_argument('--dataset', type=str, default='ycb')
parser.add_argument('--dataset_root', type=str, default='datasets/ycb/')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--lr', default=0.0001, help='learning rate', type=float)
parser.add_argument('--lr_rate', default=0.1, help='learning rate decay rate')
parser.add_argument('--warnup_iters', default=500)
parser.add_argument('--decay_epoch', type=int, default=27)
parser.add_argument('--cos', type=int, default=0, help='cosine lr schedule')
parser.add_argument('--nepoch', type=int, default=30,
                    help='max number of epochs to train')
parser.add_argument('--resume', type=str, default='',
                    help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--exp_name', type=str, default="baseline")
parser.add_argument('--outf', type=str, default='experiments/ycb/')
parser.add_argument('--log_dir', type=str, default='experiments/ycb/')
parser.add_argument('--augmask', action='store_true', help='aug mask xyz & uv')
parser.add_argument('--test', action='store_true', help='test only')
parser.add_argument('--vis', action='store_true', help='vis sampled data')
parser.add_argument('--pe', action='store_true', help='add pe encoder')
parser.add_argument('--mask_xyz_only', action='store_true',
                    help='not mask rgb')
parser.add_argument('--train_list_name', type=str,
                    default='train_list.txt', help='train list name')
parser.add_argument('--aug_rgb', action='store_true',
                    help='first random aug with rgb')
parser.add_argument('--aug_rgb_sim', action='store_true',
                    help='sim sample rgb aug')
parser.add_argument('--aug_mode', type=str, default='mask',
                    help='training phase aug mode')
parser.add_argument('--aug_xyz', action='store_true', help='aug_xyz')
parser.add_argument('--aug_xyz_sim', action='store_true', help='aug_xyz_sim')
parser.add_argument('--geo6d_mode', action='store_true', help='geo6d mode')
parser.add_argument('--geo_loss_weight', type=float, default=1.0,
                    help='geo loss weight')


opt = parser.parse_args()
st_time = time.time()


def main():
    # pre-setup
    torch.backends.cudnn.enabled = True
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    opt.gpu_number = torch.cuda.device_count()

    opt.outf = os.path.join(opt.outf, opt.exp_name)
    opt.log_dir = os.path.join(opt.log_dir, opt.exp_name)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 512
        if os.path.exists(opt.outf) == False:
            os.makedirs(opt.outf)
        if os.path.exists(opt.log_dir) == False:
            os.makedirs(opt.log_dir)

    mp.spawn(per_processor, nprocs=opt.gpu_number, args=(opt,))


def predict(data, estimator, lossor, opt, mode='train', pe=False):
    cls_ids = data['class_id'].to(opt.gpu)
    rgb = data['rgb'].to(opt.gpu)
    depth = data['xyz'].to(opt.gpu)
    mask = data['mask'].to(opt.gpu)
    gt_r = data['target_r'].to(opt.gpu)
    gt_t = data['target_t'].to(opt.gpu)

    preds = estimator(rgb, depth, cls_ids, pe)
    if opt.geo6d_mode:
        preds['delta_abc_target'] = data['delta_abc_target'].to(opt.gpu)

    loss, loss_dict = lossor(preds, mask, gt_r, gt_t, cls_ids)
    if mode == 'train':
        return loss, loss_dict

    if mode == 'test':
        preds['xyz'] = mask
        res_T = post_processing(preds, opt.sym_list)
        bs, _, _ = res_T.size()
        res_T = res_T.cpu().numpy()
        tar_T = torch.cat([gt_r, gt_t.unsqueeze(dim=2)], dim=2)
        tar_T = tar_T.cpu().numpy()

        gt_cls = data['class_id'].cpu().numpy().astype(np.int32)
        model_xyz = data['model_xyz'].cpu().numpy()

        rt_list = []
        gt_rt_list = []
        gt_cls_list = []
        model_list = []
        for i in range(bs):
            scale = opt.obj_radius[int(gt_cls[i])]
            res_T[i, :, 3] *= scale
            tar_T[i, :, 3] *= scale
            model_xyz[i] *= scale
            rt_list.append(res_T[i])
            gt_rt_list.append(tar_T[i])
            gt_cls_list.append(gt_cls[i] + 1)
            model_list.append(model_xyz[i])

        return loss, loss_dict, rt_list, gt_rt_list, gt_cls_list, model_list


def per_processor(gpu, opt):
    opt.gpu = gpu
    tensorboard_writer = 0
    if gpu == 0:
        tensorboard_writer = SummaryWriter(opt.log_dir)
    torch.distributed.init_process_group(
        backend='nccl', init_method='tcp://localhost:23456', rank=gpu, world_size=opt.gpu_number)
    print("init gps:{}".format(gpu))
    torch.cuda.set_device(gpu)

    # init DDP model
    in_channel = 6 if not opt.geo6d_mode else 10
    estimator = icp_pose_xyz.FCN6D(
        num_class=opt.num_objects, in_channel=in_channel).to(gpu)
    estimator = torch.nn.parallel.DistributedDataParallel(
        estimator, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    # init optimizer
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr * opt.gpu_number)

    # resume from existed model
    epoch = 0
    if opt.resume != '':
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(
            '{0}/{1}'.format(opt.outf, opt.resume), map_location=loc)
        model_dict = estimator.state_dict()
        same_dict = {
            k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(same_dict)
        estimator.load_state_dict(model_dict)
        print("loaded checkpoint '{}' (epoch {})".format(
            opt.resume, checkpoint['epoch']))
        epoch = checkpoint['epoch']

    # init DDP dataloader
    dataset = PoseDataset_ycb_online('train', opt, True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.workers, pin_memory=True, sampler=sampler)
    if gpu == 0:
        test_set = PoseDataset_ycb_online('test', opt, False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size // 2, shuffle=False,
                                                  num_workers=opt.workers // 2, pin_memory=True)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    opt.obj_radius = dataset.obj_radius

    # init loss model
    lossor = icp_pose_xyz.get_loss(
        dataset, geo_loss_weight=opt.geo_loss_weight).to(gpu)

    # epoch loop
    tensorboard_loss_list = []
    tensorboard_test_list = []
    start_epoch = opt.start_epoch if epoch == 0 else epoch
    for epoch in range(start_epoch, opt.nepoch):
        sampler.set_epoch(epoch)
        opt.cur_epoch = epoch

        # # train for one epoch
        print('>>>>>>>>>>>train>>>>>>>>>>>')
        if not opt.test:
            train(dataloader, estimator, lossor, optimizer, epoch,
                  tensorboard_writer, tensorboard_loss_list, opt)
            torch.cuda.empty_cache()
            # save checkpoint
            if gpu == 0:
                print('>>>>>>>>>>>save checkpoint>>>>>>>>>>')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': estimator.state_dict()},
                    '{}/checkpoint_{:04d}.pth.tar'.format(opt.outf, epoch))
        # test for one epoch
        if gpu == 0:
            print('>>>>>>>>>>>test>>>>>>>>>>>')
            test(test_loader, estimator, lossor, epoch,
                 tensorboard_writer, tensorboard_test_list, opt)
            torch.cuda.empty_cache()


def train(train_loader, estimator, lossor, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt):
    if opt.gpu == 0:
        train_loss_list = []
        logger = setup_logger('epoch%d' % epoch, os.path.join(
            opt.log_dir, 'epoch_%d_log.txt' % epoch))

    estimator.train()
    optimizer.zero_grad()

    i = 0
    for data in train_loader:
        i += 1
        # update learning rate
        iter_th = epoch * len(train_loader) + i
        cur_lr = adjust_learning_rate(optimizer, epoch, iter_th, opt)
        loss, loss_dict = predict(
            data, estimator, lossor, opt, mode='train', pe=opt.pe)
        loss.backward()
        if torch.any(torch.isnan(loss)):
            raise ValueError(f"data : {data}, loss_dict: {loss_dict}")
        optimizer.step()
        optimizer.zero_grad()

        # log and draw loss
        if opt.gpu == 0:
            train_loss_list.append(loss_dict)
            log_function(train_loss_list, logger, epoch, i, cur_lr)

            if len(train_loss_list) % 50 == 0:
                l_dict = deepcopy(train_loss_list[-50])
                for ld in train_loss_list[-49:]:
                    for key in ld:
                        l_dict[key] += ld[key]
                for key in l_dict:
                    l_dict[key] = l_dict[key] / 50.0

                tensorboard_loss_list.append(l_dict)
                draw_loss_list('train', tensorboard_loss_list,
                               tensorboard_writer)


def test(test_loader, estimator, lossor, epoch, tensorboard_writer, tensorboard_test_list, opt):
    if opt.gpu == 0:
        test_loss_list = []
        logger = setup_logger('epoch%d' % epoch, os.path.join(
            opt.log_dir, 'test_%d_log.txt' % epoch))
        # init evaluator
        ycb_evaluator = YCBEval()

    estimator.eval()

    with torch.no_grad():
        i = 0
        total_rt_list = []
        total_gt_list = []
        total_cls_list = []
        for data in test_loader:
            i += 1
            _, test_loss_dict, rt_list, gt_rt_list, gt_cls_list, model_list = predict(
                data, estimator, lossor, opt, mode='test')
            total_rt_list += rt_list
            total_gt_list += gt_rt_list
            total_cls_list += gt_cls_list

            # eval
            ycb_evaluator.eval_pose_parallel(
                rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)

            # log and draw loss
            if opt.gpu == 0:
                # log
                test_loss_list.append(test_loss_dict)
                log_function(test_loss_list, logger, epoch, i, opt.lr)

        save_pred_and_gt_json(total_rt_list, total_gt_list,
                              total_cls_list, opt.log_dir)

        # draw loss
        if opt.gpu == 0:
            # evaluation result
            cur_eval_info_dict = ycb_evaluator.cal_auc()
            # draw
            l = deepcopy(test_loss_list[0])
            for ld in test_loss_list[1:]:
                for key in ld:
                    l[key] += ld[key]
            for key in l:
                l[key] = l[key] / len(test_loss_list)

            l['auc'] = cur_eval_info_dict['auc']
            tensorboard_test_list.append(l)
            draw_loss_list('test', tensorboard_test_list, tensorboard_writer)

            # output test result
            log_tmp = 'TEST ENDING: '
            for key in l:
                log_tmp = log_tmp + ' {}:{:.4f}'.format(key, l[key])
            logger.info(log_tmp)
            for key in cur_eval_info_dict:
                log_tmp = log_tmp + \
                    ' {}:{:.2f}'.format(key, cur_eval_info_dict[key])
                logger.info(log_tmp)


def adjust_learning_rate(optimizer, epoch, iter, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr * opt.gpu_number
    if opt.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.nepoch))
    else:  # stepwise lr schedule
        lr *= opt.lr_rate if epoch >= opt.decay_epoch else 1.

    if (iter <= opt.warnup_iters):
        lr = warnup_lr(iter, opt.warnup_iters, lr / 10, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def log_function(loss_list, logger, epoch, batch, lr):
    l = loss_list[-1]
    tmp = 'time{} E{} B{} lr:{:.9f}'.format(
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, batch, lr)
    for key in l:
        tmp = tmp + ' {}:{:.4f}'.format(key, l[key])
    logger.info(tmp)


def draw_loss_list(phase, loss_list, tensorboard_writer):
    loss = loss_list[-1]
    for key in loss:
        tensorboard_writer.add_scalar(phase+'/'+key, loss[key], len(loss_list))


if __name__ == '__main__':
    main()
