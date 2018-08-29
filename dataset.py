from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.ssv2 import SSV2
from datasets.ssv1 import SSV1
from datasets.ssv2_flow import SSV2FLOW
from datasets.ucf101flow import UCF101FLOW
from enum import Enum


class Subset(Enum):
    training = 'training'
    validation = 'validation'
    test = 'test'


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    global training_data

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            Subset.training.value,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101flow':
        training_data = UCF101FLOW(
            opt.video_path,
            opt.annotation_path,
            subset='training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ssv2':
        training_data = SSV2(
            opt.video_path,
            opt.something_train_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv1':
        training_data = SSV1(
            opt.video_path,
            opt.something_train_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv2flow':
        training_data = SSV2FLOW(
            opt.video_path,
            opt.something_train_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    global validation_data

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101flow':
        validation_data = UCF101FLOW(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ssv2':
        validation_data = SSV2(
            opt.video_path,
            opt.something_val_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv1':
        validation_data = SSV1(
            opt.video_path,
            opt.something_val_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv2flow':
        validation_data = SSV2FLOW(
            opt.video_path,
            opt.something_val_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    global subset, test_data
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101flow':
        test_data = UCF101FLOW(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ssv2':
        test_data = SSV2(
            opt.video_path,
            opt.something_test_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv1':
        test_data = SSV1(
            opt.video_path,
            opt.something_test_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'ssv2flow':
        test_data = SSV2FLOW(
            opt.video_path,
            opt.something_train_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path,
            flow_x_images_path=opt.flow_x_path,
            flow_y_images_path=opt.flow_y_path
        )

    return test_data
