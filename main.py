import json
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim

from dataset import data_set
from model import generate_model
from opts import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from target_transforms import ClassLabel
from temporal_transforms import LoopPadding, TemporalRandomCrop
from train import train_epoch
from utils import Logger
from validation import val_epoch


def worker_init_fn(worker_id):
    random.seed(worker_id)


if __name__ == '__main__':
    # コマンドラインオプションを取得
    opt = parse_opts()

    # チャンネル数を取得
    opt.n_channel = 3
    if opt.add_gray_image_paths:
        opt.n_channel += len(opt.add_gray_image_paths)
    if opt.add_RGB_image_paths:
        opt.n_channel += len(opt.add_RGB_image_paths) * 3

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
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

    with open(os.path.join(result_dir_name, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    cudnn.deterministic = True

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    optimizer = None
    train_loader = None
    train_logger = None
    val_loader = None
    val_logger = None

    # 画像郡のパスとそのチャンネル数をそれぞれ辞書に登録
    paths = {opt.video_path: '3ch'}
    if opt.add_gray_image_paths:
        for one_ch in opt.add_gray_image_paths:
            paths[one_ch] = '1ch'
    if opt.add_RGB_image_paths:
        for three_ch in opt.add_RGB_image_paths:
            paths[three_ch] = '3ch'

    if not opt.no_train:
        crop_method = None
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value),
            norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = data_set[opt.data_set](
            paths, opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
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
            nesterov=opt.nesterov)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value),
            norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = data_set[opt.data_set](
            paths, opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
        )
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        if opt.show_top5:
            val_logger = Logger(
                os.path.join(result_dir_name, 'val.log'),
                ['epoch', 'loss', 'acc-top1', 'acc-top5', 'batch-time', 'epoch-time'])
        else:
            val_logger = Logger(
                os.path.join(result_dir_name, 'val.log'),
                ['epoch', 'loss', 'acc-top1', 'batch-time', 'epoch-time'])

    if opt.resume_path:
        path = os.path.join(result_dir_name, opt.resume_path)
        print('loading checkpoint {}'.format(path))
        checkpoint = torch.load(path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['lr'] = opt.learning_rate

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, result_dir_name)
        if not opt.no_val:
            val_epoch(i, val_loader, model, criterion, opt, val_logger)
