import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model import WSAVS
from datasets import get_val_test_dataset, inverse_normalize
import cv2

import torch.multiprocessing as mp
import torch.distributed as dist


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='wsavs', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_pred_masks', action='store_true', help='Set to store all pred masks (saved in pred_masks directory within experiment folder)')

    # Dataset
    parser.add_argument('--trainset', default='avsbench', type=str, help='trainset (avsbench)')
    parser.add_argument('--testset', default='avsbench', type=str, help='testset (avsbench)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # ws-avs hyper-params
    parser.add_argument('--model', default='wsavs')
    parser.add_argument('--imgnet_type', default='resnet50')
    parser.add_argument('--audnet_type', default='resnet50')

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # evaluation parameters
    parser.add_argument("--dropout_img", type=float, default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(local_rank, ngpus_per_node, args):
    args.gpu = local_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
            print(args.dist_url, args.world_size, args.rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model
    if args.model.lower() == 'wsavs':
        model = WSAVS(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, imgnet_type=args.imgnet_type, audnet_type=args.audnet_type)
    else:
        raise NotImplementedError

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.multiprocessing_distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        print('model:', ckp['model'].keys())
        if args.multiprocessing_distributed:
            model.load_state_dict(ckp['model'])
        else:
            model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    testdataset = get_val_test_dataset(args)
    if args.multiprocessing_distributed:
        sampler = torch.utils.data.DistributedSampler(testdataset, num_replicas=ngpus_per_node, rank=args.rank, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(testdataset)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    print("Loaded dataloader.")

    validate(testdataloader, model, model_dir, args)


@torch.no_grad()
def validate(test_loader, model, model_dir, args):
    model.train(False)
    miou_mtr = AverageMeter('MIOU', ':.6f')
    fscore_mtr = AverageMeter('FScore', ':.6f')

    name_list = []
    miou_list = []
    fscore_list = []

    for step, (image, spec, anno, name) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            gt_mask = anno['gt_map'].cuda(args.gpu, non_blocking=True)
        
        pred_mask = model(image.float(), spec.float(), pseudo_mask=gt_mask, mode='test')

        miou = utils.mask_iou(pred_mask, gt_mask)
        fscore = utils.Eval_Fmeasure(pred_mask, gt_mask.float(), model_dir)

        miou_mtr.update(miou.item(), image.shape[0])
        fscore_mtr.update(fscore.item(), image.shape[0])

        if args.save_pred_masks:
            mask_save_path = os.path.join(model_dir, 'pred_masks')
            utils.save_mask(pred_mask, mask_save_path, name[0])

        name_list.append(name[0])
        miou_list.append(miou)
        fscore_list.append(fscore)

    metric_save_file = os.path.join(model_dir, 'metrics.txt')
    utils.save_metrics(metric_save_file, name_list, miou_list, fscore_list)

    miou = miou_mtr.avg
    fscore = fscore_mtr.avg

    print(f'mIoU: {miou}')
    print(f'FScore: {fscore}')


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main(get_arguments())
