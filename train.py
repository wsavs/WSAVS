import os
import argparse
import builtins
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

import utils
from model import WSAVS
from datasets import get_train_dataset, get_val_test_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='wsavs', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='avsbench', type=str, help='trainset (avsbench)')
    parser.add_argument('--testset', default='avsbench', type=str, help='testset,(avsbench)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--train_pseudo_gt_path', default='', type=str, help='Pseudo mask directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str, help='Ground-truth directory path of test data')

    # ws-avs hyper-params
    parser.add_argument('--model', default='wsavs')
    parser.add_argument('--imgnet_type', default='resnet50')
    parser.add_argument('--audnet_type', default='resnet50')
    parser.add_argument("--avfusion_stages", default=[0,1,2,3], nargs='+', type=int, help='avfusion stages: 0, 1, 2, 3')

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--lr_schedule", default='cte', help="learning rate schedule")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument("--clip_norm", type=float, default=0, help="gradient clip norm")
    parser.add_argument("--dropout_img", type=float, default=0.9, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    parser.add_argument("--weight_msmil", type=float, default=1., help="weight for msmil loss")
    parser.add_argument("--weight_pixel", type=float, default=1., help="dropout for pixel loss")

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


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # tb writers
    tb_writer = SummaryWriter(model_dir)

    # logger
    log_fn = f"{model_dir}/train.log"
    def print_and_log(*content, **kwargs):
        # suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(log_fn, 'a') as f:
            f.write(msg+'\n')
    builtins.print = print_and_log

    # Create model
    if args.model.lower() == 'wsavs':
        model = WSAVS(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, imgnet_type=args.imgnet_type, audnet_type=args.audnet_type, avfusion_stages=args.avfusion_stages)
    else:
        raise NotImplementedError
        
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_mIoU, best_FScore = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch, best_mIoU, best_FScore = ckp['epoch'], ckp['best_mIoU'], ckp['best_FScore']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Dataloaders
    traindataset = get_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    valdataset = get_val_test_dataset(args)
    val_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    mIoU, FScore = validate(val_loader, model, args)
    print(f'mIoU (epoch {start_epoch}): {mIoU}')
    print(f'FScore (epoch {start_epoch}): {FScore}')
    print(f'best_mIoU: {best_mIoU}')
    print(f'best_FScore: {best_FScore}')

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args, tb_writer)

        # Evaluate
        mIoU, FScore = validate(val_loader, model, args)
        if mIoU >= best_mIoU:
            best_mIoU, best_FScore = mIoU, FScore
        print(f'mIoU (epoch {epoch+1}): {mIoU}')
        print(f'FScore (epoch {epoch+1}): {FScore}')
        print(f'best_mIoU: {best_mIoU}')
        print(f'best_FScore: {best_FScore}')

        tb_writer.add_scalar('mIoU', mIoU, epoch)
        tb_writer.add_scalar('FScore', FScore, epoch)

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_mIoU': best_mIoU,
                   'best_FScore': best_FScore}
            torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
            if mIoU == best_mIoU:
                torch.save(ckp, os.path.join(model_dir, 'best.pth'))
            print(f"Model saved to {model_dir}")


def train(train_loader, model, optimizer, epoch, args, writer):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')
    msmil_loss_mtr = AverageMeter('MSMIL Loss', ':.3f')
    pixel_loss_mtr = AverageMeter('PIXEL Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr, msmil_loss_mtr, pixel_loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, spec, anno, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step = i + len(train_loader) * epoch
        utils.adjust_learning_rate(optimizer, epoch + i / len(train_loader), args)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            pseudo_mask = anno['gt_map'].cuda(args.gpu, non_blocking=True)

        msmil_loss, pixel_loss, _ = model(image.float(), spec.float(), pseudo_mask=pseudo_mask)
        loss = args.weight_msmil * msmil_loss + args.weight_pixel * pixel_loss

        loss_mtr.update(loss.item(), image.shape[0])
        msmil_loss_mtr.update(msmil_loss.item(), image.shape[0])
        pixel_loss_mtr.update(pixel_loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        if args.clip_norm != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # clip gradient

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('loss', loss_mtr.avg, global_step)
        writer.add_scalar('msmil_loss', msmil_loss_mtr.avg, global_step)
        writer.add_scalar('pixel_loss', pixel_loss_mtr.avg, global_step)

        # writer.add_scalar('batch_time', batch_time.avg, global_step)
        # writer.add_scalar('data_time', data_time.avg, global_step)

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        del loss


def validate(test_loader, model, args):
    model.train(False)
    miou_mtr = AverageMeter('MIOU', ':.6f')
    fscore_mtr = AverageMeter('FScore', ':.6f')
    for step, (image, spec, anno, name) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            # label = anno['class'].cuda(args.gpu, non_blocking=True)
            gt_mask = anno['gt_map'].cuda(args.gpu, non_blocking=True)
        
        pred_mask = model(image.float(), spec.float(), pseudo_mask=gt_mask, mode='test')

        miou = utils.mask_iou(pred_mask, gt_mask)
        fscore = utils.Eval_Fmeasure(pred_mask, gt_mask.float(), os.path.join(args.model_dir, args.experiment_name))

        miou_mtr.update(miou.item(), image.shape[0])
        fscore_mtr.update(fscore.item(), image.shape[0])
    
    miou = miou_mtr.avg
    fscore = fscore_mtr.avg

    return miou, fscore


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
