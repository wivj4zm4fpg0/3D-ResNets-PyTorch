import time

import torch

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_1_5


def val_epoch(epoch, data_loader, model, criterion, opt, epoch_logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

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
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
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
            'batch': opt.batch_size,
            'batch-time': batch_time.avg,
            'epoch-time': epoch_time
        })
    else:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc-top1': accuracies.avg,
            'batch': opt.batch_size,
            'batch-time': batch_time.avg,
            'epoch-time': epoch_time
        })

    return losses.avg
