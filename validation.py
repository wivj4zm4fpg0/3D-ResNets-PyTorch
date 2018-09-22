import time

import torch

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies5 = AverageMeter()

    end_time = time.time()
    epoch_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc1, inputs.size(0))
            accuracies5.update(acc5, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc-Top1 {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Acc-Top5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies,
                acc5=accuracies5))

    epoch_time = time.time() - epoch_time

    logger.log({'epoch': epoch,
                'loss': losses.avg,
                'acc-top1': accuracies.avg,
                'acc-top5': accuracies5.avg,
                'time': epoch_time})

    return losses.avg
