import os
import time
from argparse import Namespace

import torch
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer as torch_optimizer
from torch.utils.data import DataLoader

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_1_5, Logger


def train_epoch(
        epoch: int,
        data_loader: DataLoader,
        model: torch.nn.Module,
        criterion: CrossEntropyLoss,
        optimizer: torch_optimizer,
        opt: Namespace,
        epoch_logger: Logger,
        result_dir_name: str
):
    print(f'train at epoch {epoch}')

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    if opt.show_top5:
        accuracies5 = AverageMeter()
    else:
        accuracies5 = None

    end_time = time.time()
    epoch_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        if opt.show_top5:
            acc1, acc5 = calculate_accuracy_1_5(outputs, targets)
            accuracies.update(acc1, inputs.size(0))
            accuracies5.update(acc5, inputs.size(0))
        else:
            acc1 = calculate_accuracy(outputs, targets)
            accuracies.update(acc1, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if opt.show_top5:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc-Top1 {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Acc-Top5 {acc5.val:.3f} ({acc5.avg:.3f))\t'
                  .format(epoch, i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies,
                          acc5=accuracies5))
        else:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc-Top1 {acc.val:.3f} ({acc.avg:.3f})\t'
                  .format(epoch, i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))

    epoch_time = time.time() - epoch_time

    if opt.show_top5:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc-top1': accuracies.avg,
            'acc-top5': accuracies5.avg,
            'lr': optimizer.param_groups[0]['lr'],
            'batch': opt.batch_size,
            'batch-time': batch_time.avg,
            'epoch-time': epoch_time
        })
    else:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc-top1': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr'],
            'batch': opt.batch_size,
            'batch-time': batch_time.avg,
            'epoch-time': epoch_time
        })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(result_dir_name, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
