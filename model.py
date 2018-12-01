import copy

import torch
from torch import nn
from torch.nn import init

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(opt):
    global model

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                n_channel=opt.n_channel)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # Heの初期値で初期化
    if not opt.resume_path:
        for module in model.modules():
            if hasattr(module, 'weight'):
                if not ('Norm' in module.__class__.__name__):
                    init.kaiming_uniform_(module.weight, mode='fan_out')
                else:
                    init.constant_(module.weight, 1)
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    if opt.pre_train_path or opt.show_answer_pre_train_model_path:
        print('loading pre-trained model {}'.format(opt.pre_train_path))
        pre_train = torch.load(opt.pre_train_path)

        if opt.n_channel != 3:
            # RGB画像のみで学習済みのモデルを転用するとき4チャンネル以降をRGBの平均にする（最初のCNN層のみ）
            temp = copy.copy(pre_train['state_dict']['module.conv1.weight'])

            pre_train['state_dict']['module.conv1.weight'] = nn.Conv3d(
                opt.n_channel,
                64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=(3, 3, 3),
                bias=False
            ).weight
            new_conv = pre_train['state_dict']['module.conv1.weight'].data

            temp_input_channel_length = len(temp.data[0])
            new_conv_input_channel_length = len(new_conv[0])
            subtraction_len = new_conv_input_channel_length - temp_input_channel_length
            output_channel_length = len(temp.data)

            # チャンネル数が3より大きい場合は4以降を3チャンネルの平均にする
            if opt.n_channel > 3:
                for i in range(output_channel_length):
                    for j in range(temp_input_channel_length):
                        new_conv[i][j] = temp.data[i][j]
                    avg = torch.sum(temp.data[i], 0) / 3
                    for j in range(subtraction_len):
                        new_conv[i][temp_input_channel_length + j] = avg
            # チャンネル数が3より小さい場合は全部3チャンネルの平均にする
            elif opt.n_channel < 3:
                for i in range(output_channel_length):
                    avg = torch.sum(temp.data[i], 0) / 3
                    for j in range(new_conv_input_channel_length):
                        new_conv[i][j] = avg

        model.load_state_dict(pre_train['state_dict'])

        if opt.model == 'densenet':
            model.module.classifier = nn.Linear(
                model.module.classifier.in_features, opt.n_fine_tune_classes)
            if not opt.no_cuda:
                model.module.classifier = model.module.classifier.cuda()
        else:
            # 転移学習をするときは全結合層以外のパラメータを更新しないようにする
            if opt.transfer_learning:
                for p in model.parameters():
                    p.requires_grad = False

            model.module.fc = nn.Linear(model.module.fc.in_features,
                                        opt.n_fine_tune_classes)
            if not opt.no_cuda:
                model.module.fc = model.module.fc.cuda()

        if opt.transfer_learning:
            parameters = model.module.fc.parameters()
        else:
            parameters = model.parameters()
        return model, parameters

    return model, model.parameters()
