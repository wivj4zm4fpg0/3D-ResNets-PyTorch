import random

import torch.utils.data
from torch import nn
from torch import optim
from torch.backends import cudnn
from torchvision import transforms

from dataset import data_set
from model import generate_model
from opts import parse_opts
from show_answer import show_answer_epoch
from target_transforms import ClassLabel


def worker_init_fn(worker_id: int):
    random.seed(worker_id)


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

random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if not opt.no_cuda:
    cudnn.benchmark = True
    cudnn.deterministic = True

model, parameters = generate_model(opt)
print(model)
criterion = nn.CrossEntropyLoss()
if not opt.no_cuda:
    criterion = criterion.cuda()

optimizer = None
train_loader = None
val_loader = None

# 画像郡のパスとそのチャンネル数をそれぞれ辞書に登録
paths = {opt.video_path: '3ch'}
if opt.add_gray_image_paths:
    for one_ch in opt.add_gray_image_paths:
        paths[one_ch] = '1ch'
if opt.add_RGB_image_paths:
    for three_ch in opt.add_RGB_image_paths:
        paths[three_ch] = '3ch'

spatial_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])
target_transform = ClassLabel(True)
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

dampening = 0 if opt.nesterov else opt.dampening
optimizer = optim.SGD(
    model.parameters(),
    lr=opt.learning_rate,
    momentum=opt.momentum,
    dampening=dampening,
    weight_decay=opt.weight_decay,
    nesterov=opt.nesterov
)
spatial_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])
target_transform = ClassLabel(True)
validation_data = data_set[opt.data_set](
    paths,
    opt.annotation_path,
    'validation',
    opt.n_val_samples,
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

if opt.show_answer_resume_path:
    print('loading checkpoint {}'.format(opt.show_answer_resume_path))
    checkpoint = torch.load(opt.show_answer_resume_path)
    assert opt.arch == checkpoint['arch']

    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if not opt.no_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = opt.learning_rate

with open(opt.show_answer_result_path, 'w') as f:
    f.write('video_name model_answer true_answer answer subset\n')
print('run')
show_answer_epoch(train_loader, model, opt, 'training')
show_answer_epoch(val_loader, model, opt, 'validation')
