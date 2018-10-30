import time

import torch

from utils import AverageMeter, image_show_calculate_accuracy


def image_show_epoch(epoch, data_loader, model, opt):
    print('image show at epoch {}'.format(epoch))

    model.eval()

    data_time = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets, targets_name) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            outputs = model(inputs)
            image_show_calculate_accuracy(outputs, targets, targets_name)
