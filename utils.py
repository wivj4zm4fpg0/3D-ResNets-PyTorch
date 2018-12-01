import csv
import os
from torch import tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path: str, header: list):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        if os.path.getsize(path) == 0:  # confirm file size
            self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values: dict):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path: str) -> float:
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs: tensor, targets: tensor) -> float:
    batch_size = targets.size(0)
    _, prediction = outputs.topk(1, 1, True)
    prediction = prediction.t()
    correct = prediction.eq(targets.view(1, -1))
    n_correct_elem = correct.sum().item()

    return n_correct_elem / batch_size


def calculate_accuracy_1_5(outputs: tensor, targets: tensor) -> tuple:
    batch_size = targets.size(0)
    _, prediction = outputs.topk(5, 1, True)
    prediction = prediction.t()
    correct = prediction.eq(targets.view(1, -1))
    n_correct_elms_1 = correct[0].float().sum().item()
    n_correct_elms_5 = correct.float().sum().item()

    return n_correct_elms_1 / batch_size, n_correct_elms_5 / batch_size
