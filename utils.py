import os
import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        if os.path.getsize(path) == 0:  # confirm file size
            self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(5, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems1 = correct[0].float().sum().item()
    n_correct_elems5 = correct.float().sum().item()

    return n_correct_elems1 / batch_size, n_correct_elems5 / batch_size
