import time
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from utils import AverageMeter


def show_answer_epoch(
        data_loader: DataLoader,
        model: torch.nn.Module,
        opt: Namespace,
        subset: str
):
    model.eval()

    data_time = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for inputs, targets, targets_name in data_loader:
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            outputs = model(inputs)
            show_answer_calculate_accuracy(outputs, targets, targets_name, opt, subset)


def show_answer_calculate_accuracy(
        outputs: torch.tensor,
        targets: torch.tensor,
        targets_name: torch.tensor,
        opt: Namespace,
        subset: str
):
    _, prediction = outputs.topk(1, 1, True)
    prediction = prediction.t()
    targets = targets.view(1, -1)
    correct = prediction.eq(targets)
    with open(opt.show_answer_result_path, 'a') as f:
        for i in range(len(correct[0])):
            model_answer = 'no_action' if prediction[0][i].item() == 0 else 'action'
            true_answer = 'no_action' if targets[0][i].item() == 0 else 'action'
            answer = 'miss' if correct[0][i].item() == 0 else 'correct'
            print('{}, model={}, true={}, {}'.format(
                targets_name[i],
                model_answer,
                true_answer,
                answer
            ))
            f.write('{} {} {} {} {}\n'.format(
                targets_name[i],
                model_answer,
                true_answer,
                answer,
                subset
            ))
