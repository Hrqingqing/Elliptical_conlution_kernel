import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dataset
import resnet18

parser = argparse.ArgumentParser(description='Propert ResNet18 for ImageNet in pytorch')
parser.add_argument('--dataset', default='imagenet', help='cifar10 or cifar100 ')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ngpus', default=4, type=int,
                    help='number of gpus')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--kernel_shape', default='only_ellipse', type=str,
                    help='kernel_shape:only_ellipse,square_ellipse,both')
global args
args = parser.parse_args()


def get_kernel_shape():
    return args.kernel_shape

def main():
    top1_old_file = 0
    best_prec1 = 0
    top5_old_file = 0
    best_prec5 = 0
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model = resnet18.resnet18()
    model = torch.nn.DataParallel(model, device_ids=range(args.ngpus))
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset == 'imagenet':
        # train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
        train_loader, test_loader = dataset.getImageNet(batch_size=args.batch_size, num_workers=1)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 60, 90], last_epoch=args.start_epoch - 1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    #                                                           verbose=True,
    #                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                                           min_lr=0, eps=1e-08)

    if args.evaluate:
        validate(test_loader, model, criterion)
        return
    if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
        os.makedirs(os.path.join(args.save_dir, args.dataset))
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, prec5 = validate(test_loader, model, criterion)

        if prec1 > best_prec1:
            top1_new_file = os.path.join(
                os.path.join(args.save_dir, args.dataset),
                'model_{}_top1_{}.th'.format(epoch,
                                             str(prec1)))
            model_snapshot(model, top1_new_file, old_file=top1_old_file, verbose=True)
            best_prec1 = prec1
            top1_old_file = top1_new_file
        if prec5 > best_prec5:
            top5_new_file = os.path.join(
                os.path.join(args.save_dir, args.dataset),
                'model_{}_top5_{}.th'.format(epoch,
                                             str(prec5)))
            model_snapshot(model, top5_new_file, old_file=top5_old_file, verbose=True)
            best_prec5 = prec5
            top5_old_file = top5_new_file
        test_file = f"resnet18-imagenet.txt"
        with open(test_file, "a+") as f:
            f.write(
                "dataset: {} \t best_acc_top1: {} \t best_acc_top5: {} \n".format(args.dataset,
                                                                                                      best_prec1,
                                                                                                      best_prec5))


train_all_info = f"./results/train_all_info.txt"
train_partial_info = f"./results/train_partial_info.txt"


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    loss_sum = []
    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))

        loss_sum.append(losses.val)
        top1.update(prec1.item(), input.size(0))

        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Train Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                  'Train Data {data_time.val:.3f} (avg: {data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Train Top1 Acc {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                  'Train Top5 Acc {top5.val:.3f} (avg: {top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        with open(train_all_info, "a+") as f:
            f.write(f"train_loss: {losses.val:.4f}\ttop1_acc: {top1.avg:.3f}\ttop5_acc: {top5.avg:.3f} \n")
    with open(train_partial_info, "a+") as f:
        f.write(
            f"train_loss: {sum(loss_sum) / len(loss_sum):.4f}\ttop1_acc: {top1.avg:.3f}\ttop5_acc: {top5.avg:.3f} \n")


test_all_info = f"./results/test_all_info.txt"
test_partial_info = f"./results/test_partial_info.txt"


def validate(test_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    test_loss_sum = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))

            test_loss_sum.append(losses.val)
            top1.update(prec1.item(), input.size(0))

            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Test Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Test Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                      'Test Top1 Acc {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                      'Test Top5 Acc {top5.val:.3f} (avg: {top5.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            with open(test_all_info, "a+") as f:
                f.write(f"test_loss: {losses.val:.4f}\t top1_acc: {top1.avg:.3f}\ttop5_acc: {top5.avg:.3f} \n")
        with open(test_partial_info, "a+") as f:
            f.write(
                f"test_loss: {sum(test_loss_sum) / len(test_loss_sum):.4f}\t top1_acc: {top1.avg:.3f}\ttop5_acc: {top5.avg:.3f} \n")

    print(' * Prec@1 {top1.avg:.3f} \t Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))


def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


if __name__ == '__main__':
    main()
