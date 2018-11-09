import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='this is train and test program by 3D-CNN model')
    parser.add_argument(
        '--video_path',
        default=None,
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default=None,
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--data_set',
        default=None,
        type=str,
        choices=['ucf101', 'shoplifting'],
        help='Used data_set (ucf101 | shoplifting)'
    )
    parser.add_argument(
        '--n_classes',
        default=None,
        type=int,
        help='Number of classes (ucf101: 101, shoplifting: 2)'
    )
    parser.add_argument(
        '--n_fine_tune_classes',
        default=None,
        type=int,
        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multi-scale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multi-scale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multi-scale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        choices=['random', 'corner', 'center'],
        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center. '
             ' (random | corner | center)'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--learning_rate_schedule', default=None, type=dict,
        help='setting learning rate per epoch. example:{1:0.001, 15:0.0001} ({epoch number:learning rate})'
    )  # TODO implement
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default=None,
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pre_train_path', default=None, type=str, help='Pre-trained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=1,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_horizontal_flip',
        action='store_true',
        help='If true horizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default=None,
        type=str,
        choices=['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'],
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=None,
        type=int,
        choices=[10, 18, 34, 50, 101],
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default=None,
        type=str,
        choices=['A', 'B'],
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=None,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--transfer_learning', action='store_true', help='transfer learning by something-something'
    )
    parser.set_defaults(transfer_learning=False)
    parser.add_argument(
        '--suffix', default=None, type=str, help='suffix in made output directly'
    )
    parser.add_argument(
        '--add_gray_image_paths', default=None, nargs='*', help='channel image to add to RGB image'
    )
    parser.add_argument(
        '--add_RGB_image_paths', default=None, nargs='*', help='channel RGB image to add to RGB image'
    )
    parser.add_argument(
        '--show_answer_train', action='store_true'
    )
    parser.set_defaults(image_show_train=False)
    parser.add_argument(
        '--show_answer_validation', action='store_true'
    )
    parser.set_defaults(image_show_validation=False)
    parser.add_argument(
        '--show_answer_result_path', default='result_path', type=str
    )
    parser.add_argument(
        '--show_answer_pre_train_model_path', default=None, type=str
    )

    args = parser.parse_args()

    return args
