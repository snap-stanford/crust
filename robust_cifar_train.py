import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import copy
from torch.utils.tensorboard import SummaryWriter
#import wandb
from torch.autograd import grad
from fl_cifar import FacilityLocationCIFAR
from lazyGreedy import lazy_greedy_heap
from utils import *
from mislabel_cifar import MISLABELCIFAR10, MISLABELCIFAR100

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--exp-str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--use_crust', action='store_true',
                    help="Whether to use clusters in dataset.")
parser.add_argument('--r', type=float, default=2.0,
                    help="Distance threshold (i.e. radius) in calculating clusters.")
parser.add_argument('--fl-ratio', type=float, default=0.5,
                    help="Ratio for number of facilities.")
parser.add_argument('--mislabel-type', type=str, default='agnostic')
parser.add_argument('--mislabel-ratio', type=float, default=0.5)
parser.add_argument('--rand-number', type=int, default=0,
                    help="Ratio for number of facilities.")
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.use_crust:
        args.store_name = '_'.join([args.dataset, args.arch, args.mislabel_type, str(args.mislabel_ratio), str(args.fl_ratio), str(args.r), args.exp_str])
    else:
        args.store_name = '_'.join([args.dataset, args.arch, args.mislabel_type, str(args.mislabel_ratio), args.exp_str])
    prepare_folders(args)
    #wandb.init(project="robust_cifar", tensorboard=True, name=args.store_name)
    #wandb.config.update(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    args.num_classes = 100 if args.dataset == 'cifar100' else 10
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = MISLABELCIFAR10(root='./data', mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio, transform=transform_train, rand_number=args.rand_number, download=True)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = MISLABELCIFAR100(root='./data', mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio, transform=transform_train, rand_number=args.rand_number, download=True)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)

    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    trainval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[80, 100], last_epoch=args.start_epoch - 1)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    weights = [1] * len(train_dataset)
    weights = torch.FloatTensor(weights)
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.use_crust and epoch >= 5:
            train_dataset.switch_data()         
            # FL part
            grads_all, labels = estimate_grads(trainval_loader, model, criterion, args, epoch, log_training)
            # per-class clustering
            ssets = []
            weights = []
            for c in range(args.num_classes):
                sample_ids = np.where((labels == c) == True)[0]
                grads = grads_all[sample_ids]
                
                dists = pairwise_distances(grads)
                weight = np.sum(dists < args.r, axis=1)
                V = range(len(grads))
                F = FacilityLocationCIFAR(V, D=dists)
                B = int(args.fl_ratio * len(grads))
                sset, vals = lazy_greedy_heap(F, V, B)
                weights.extend(weight[sset].tolist())
                sset = sample_ids[np.array(sset)]
                ssets += list(sset)
            weights = torch.FloatTensor(weights)
            train_dataset.adjust_base_indx_tmp(ssets)
            label_acc = train_dataset.estimate_label_acc()
            tf_writer.add_scalar('label_acc', label_acc, epoch)
            log_training.write('epoch %d label acc: %f\n'%(epoch, label_acc))
            print("change train loader")
            
        # train for one epoch
        if args.use_crust and epoch > 5:
            train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=True)
        else:
            train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=False)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_training, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        lr_scheduler.step()

    print('best_acc1: {:.4f}'.format(best_acc1.item()))


def train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        input, target, target_real, index = batch
        if fetch:
            input_b = train_loader.dataset.fetch(target)
            lam = np.random.beta(1, 0.1)
            input = lam * input + (1 - lam) * input_b
        c_weights = weights[index]
        c_weights = c_weights.type(torch.FloatTensor)
        c_weights =  c_weights / c_weights.sum()
        if args.gpu is not None:
            c_weights = c_weights.to(args.gpu, non_blocking=True)
    
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.type(torch.FloatTensor)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, feats = model(input)
        loss = criterion(output, target)
        loss = (loss * c_weights).sum()
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log_training=None, tf_writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.type(torch.FloatTensor)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, feats = model(input)
            loss = criterion(output, target)
            loss = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
            log_training.write('epoch %d val acc: %f\n'%(epoch, top1.avg))

    return top1.avg

def estimate_grads(trainval_loader, model, criterion, args, epoch, log_training):
    # switch to train mode
    model.train()
    all_grads = []
    all_targets = []
    all_preds = []
    top1 = AverageMeter('Acc@1', ':6.2f')
    for i, (input, target, target_real, idx) in enumerate(trainval_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        all_targets.append(target)
        target = target.cuda(args.gpu, non_blocking=True)
        target_real = target_real.cuda(args.gpu, non_blocking=True)
        # compute output
        output, feat = model(input)
        _, pred = torch.max(output, 1)
        loss = criterion(output, target).mean()
        acc1, acc5 = accuracy(output, target_real, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())
    all_grads = np.vstack(all_grads)
    all_targets = np.hstack(all_targets)
    all_preds = np.hstack(all_preds)
    log_training.write('epoch %d train acc: %f\n'%(epoch, top1.avg))
    return all_grads, all_targets

if __name__ == '__main__':
    main()
