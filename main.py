import copy
import json
import os

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import test
from dataset import datasets
from mean import get_mean, get_std
from model import generate_model
from opts import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from target_transforms import ClassLabel, VideoID
from temporal_transforms import LoopPadding, TemporalRandomCrop
from train import train_epoch
from utils import Logger
from validation import val_epoch

if __name__ == '__main__':
    opt = parse_opts()
    n_channel = 3
    use_fine_tune = opt.n_finetune_classes and opt.pretrain_path
    if opt.add_image_paths:
        n_channel = n_channel + len(opt.add_image_paths)
    result_dir_name = '{}-{}-{}-{}ch'.format(opt.dataset, opt.model, opt.model_depth, n_channel)
    if opt.transfer_learning:
        result_dir_name = result_dir_name + '-transfer-learning'
    elif opt.n_finetune_classes:
        result_dir_name = result_dir_name + '-finetune-pretrain'
    else:
        result_dir_name = result_dir_name + '-no-pretrain'
    if opt.suffix:
        result_dir_name = result_dir_name + '-{}'.format(opt.suffix)
    result_dir_name = os.path.join(opt.result_path, result_dir_name)
    os.makedirs(result_dir_name, exist_ok=True)
    opt.scales = [opt.initial_scale]  # initial_scale = 1.0
    for i in range(1, opt.n_scales):  # n_scales = 5
        opt.scales.append(opt.scales[-1] * opt.scale_step)  # scale_step = 0.84089641525
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(result_dir_name, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    optimizer = None
    scheduler = None
    train_loader = None
    train_logger = None
    val_loader = None
    val_logger = None

    paths = [opt.video_path]
    if opt.add_image_paths:
        paths.extend(opt.add_image_paths)

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
            ToTensor(opt.norm_value), norm_method  # norm_value = 1
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = datasets[opt.dataset](paths, opt.annotation_path, 'training',
                                              spatial_transform=spatial_transform,
                                              temporal_transform=temporal_transform, target_transform=target_transform,
                                              n_channel=n_channel)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(result_dir_name, 'train.log'),
            ['epoch', 'loss', 'acc-top1', 'acc-top5', 'lr', 'batch'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([  # sample_size = 112
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = datasets[opt.dataset](paths, opt.annotation_path, 'validation',
                                                opt.n_val_samples,
                                                spatial_transform=spatial_transform,
                                                temporal_transform=temporal_transform,
                                                target_transform=target_transform,
                                                n_channel=n_channel)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(result_dir_name, 'val.log'), ['epoch', 'loss', 'acc-top1', 'acc-top5'])

    if opt.add_image_paths:
        temp = copy.copy(model.module.conv1)
        model.module.conv1 = nn.Conv3d(
            n_channel,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        if not opt.resume_path and use_fine_tune:
            temp_len = len(temp.weight.data[0])
            out_len = len(model.module.conv1.weight.data[0])
            sub_len = out_len - temp_len
            for i in range(len(temp.weight.data)):
                for j in range(temp_len):
                    model.module.conv1.weight.data[i][j] = temp.weight.data[i][j]
                avg = torch.sum(temp.weight.data[i], 0) / 3
                for j in range(sub_len):
                    model.module.conv1.weight.data[i][temp_len + j] = avg
        model.cuda()

    if opt.resume_path:
        opt.resume_path = os.path.join(result_dir_name, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['lr'] = opt.learning_rate

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        # validation_loss = None
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, result_dir_name)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
        # if not opt.no_train and not opt.no_val:
        #    scheduler.step(validation_loss)
    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = datasets[opt.dataset](paths, opt.annotation.path, 'test',
                                          0,
                                          sapatial_transform=spatial_transform,
                                          temporal_transform=temporal_transform,
                                          target_transform=target_transform,
                                          n_channel=n_channel)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
