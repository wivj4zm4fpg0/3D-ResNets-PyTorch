import json
import os
import random

import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.backends import cudnn
from torchvision import transforms

from dataset import data_set
from model import generate_model
from opts import parse_opts
from target_transforms import ClassLabel
from train import train_epoch
from utils import Logger
from validation import val_epoch


def worker_init_fn(worker_id: int):
    random.seed(worker_id)


# コマンドラインオプションを取得
opt = parse_opts()

device = torch.device('cpu' if opt.no_cuda else 'cuda')

# 画像郡のパスとそのチャンネル数をそれぞれ辞書に登録
paths = {}
if opt.add_gray_image_paths:
    for one_ch in opt.add_gray_image_paths:
        paths[one_ch] = '1ch'
if opt.add_RGB_image_paths:
    for three_ch in opt.add_RGB_image_paths:
        paths[three_ch] = '3ch'

# チャンネル数を取得
opt.n_channel = 0
if opt.add_gray_image_paths:
    opt.n_channel += len(opt.add_gray_image_paths)
if opt.add_RGB_image_paths:
    opt.n_channel += len(opt.add_RGB_image_paths) * 3

# ニューラルネットワークのアーキテクチャを取得
opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
# コマンドライン引数を表示
print(opt)

# 結果の出力ディレクトリの名前を自動で決める
result_dir_name = '{}-{}-{}-{}ch-{}frame-{}batch-{}size'.format(
    opt.data_set,
    opt.model,
    opt.model_depth,
    opt.n_channel,
    opt.sample_duration,
    opt.batch_size,
    opt.sample_size
)
if opt.transfer_learning:
    result_dir_name = result_dir_name + '-transfer_learning'
elif opt.n_fine_tune_classes:
    result_dir_name = result_dir_name + '-fine_tune_pre_train'
else:
    result_dir_name = result_dir_name + '-no_pre_train'
if opt.suffix:
    result_dir_name = result_dir_name + '-{}'.format(opt.suffix)
result_dir_name = os.path.join(opt.result_path, result_dir_name)
os.makedirs(result_dir_name, exist_ok=True)  # 出力ディレクトリを作成

# コマンドライン引数を保存する
with open(os.path.join(result_dir_name, 'opts.json'), 'w') as opt_file:
    json.dump(vars(opt), opt_file)

# 乱数の初期化
random.seed(opt.manual_seed)
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
# CUDAが使えるなら使う
if not opt.no_cuda:
    # この命令はモデルへの入力サイズが同じ時にプログラムを最適化できる
    cudnn.benchmark = True
    # CUDAの乱数を決定する?
    cudnn.deterministic = True

# モデルとモデルのパラメータを取得
model, parameters = generate_model(opt)
# モデルのアーキテクチャを出力
print(model)
# 損失関数をクロスエントロピーにする
criterion = nn.CrossEntropyLoss()
# CUDAが使えるなら使う
if not opt.no_cuda:
    criterion = criterion.cuda()

optimizer = None
train_loader = None
train_logger = None
val_loader = None
val_logger = None

if not opt.no_train:
    spatial_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])
    target_transform = ClassLabel()
    training_data = data_set[opt.data_set](
        paths,
        opt.annotation_path,
        'training',
        spatial_transform=spatial_transform,
        target_transform=target_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    if opt.show_top5:
        train_logger = Logger(
            os.path.join(result_dir_name, 'train.log'),
            [
                'epoch',
                'loss',
                'acc-top1',
                'acc-top5',
                'lr', 'batch',
                'batch-time',
                'epoch-time'
            ]
        )
    else:
        train_logger = Logger(
            os.path.join(result_dir_name, 'train.log'),
            [
                'epoch',
                'loss',
                'acc-top1',
                'lr', 'batch',
                'batch-time',
                'epoch-time'
            ]
        )

    dampening = 0 if opt.nesterov else opt.dampening
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov
    )

if not opt.no_val:
    spatial_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])
    target_transform = ClassLabel()
    validation_data = data_set[opt.data_set](
        paths,
        opt.annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        target_transform=target_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    if opt.show_top5:
        val_logger = Logger(
            os.path.join(result_dir_name, 'val.log'),
            ['epoch', 'loss', 'acc-top1', 'acc-top5', 'batch-time', 'epoch-time']
        )
    else:
        val_logger = Logger(
            os.path.join(result_dir_name, 'val.log'),
            ['epoch', 'loss', 'acc-top1', 'batch-time', 'epoch-time']
        )

# 重みを保存したファイルがあるなら読み込む
if opt.resume_path:
    path = os.path.join(result_dir_name, opt.resume_path)
    print('loading checkpoint {}'.format(path))
    checkpoint = torch.load(path)
    # アーキテクチャが同じかどうかチェック
    assert opt.arch == checkpoint['arch']
    # 前回のエポック数を取得
    opt.begin_epoch = checkpoint['epoch']
    # パラメータを読み込む
    model.load_state_dict(checkpoint['state_dict'])
    if not opt.no_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = opt.learning_rate

print('run')
for i in range(opt.begin_epoch, opt.n_epochs + 1):
    if not opt.no_train:
        train_epoch(i, train_loader, model, criterion, optimizer, opt,
                    train_logger, result_dir_name, device)
    if not opt.no_val:
        val_epoch(i, val_loader, model, criterion, opt, val_logger, device)
