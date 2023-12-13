import os
import json
import math

import torch
from torch.optim import *
import numpy as np

from PIL import Image


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = None
    return optimizer, scheduler


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    wu = 0 if 'warmup_epochs' not in vars(args) else args.warmup_epochs
    if args.lr_schedule == 'cos':  # cosine lr schedule
        if epoch < wu:
            lr = args.init_lr * epoch / wu
        else:
            lr = args.init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - wu) / (args.epochs - wu)))
    elif args.lr_schedule == 'cte':  # constant lr
        lr = args.init_lr
    else:
        raise ValueError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    # temp_pred = torch.sigmoid(pred)
    pred = (pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N

    return iou


def eval_pr(y_pred, y, num, cuda_flag=True):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path, pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    # pred = torch.sigmoid(pred) # =======================================[important]
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = eval_pr(pred[img_id], gt[img_id], pr_num)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0 # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    fLog.close()

    return score.max()


def save_mask(pred_mask, save_base_path, name):
    # pred_mask: [1, 224, 224]

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_mask = pred_mask.squeeze(0)
    pred_mask = (pred_mask > 0.5).int()
    
    pred_mask = pred_mask.cpu().data.numpy().astype(np.uint8)
    pred_mask *= 255

    output_name = f'{str(name)}.png'
    img = Image.fromarray(pred_mask).convert('P')
    img.save(os.path.join(save_base_path, output_name), format='PNG')


def save_metrics(filename, name_list, miou_list, fscore_list):
    with open(filename, "w") as file_iou:
        file_iou.write('name,miou,fscore\n')
        for indice in np.argsort(miou_list)[::-1]:
            file_iou.write(f"{name_list[indice]},{miou_list[indice]},{fscore_list[indice]}\n")
