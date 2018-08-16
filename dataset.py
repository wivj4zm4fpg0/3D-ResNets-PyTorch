from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.ssv2 import SSV2
from datasets.ssv1 import SSV1


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    global training_data
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'something-something-v2',
                           'something-something-v1']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'training',
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
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'something-something-v2':
        training_data = SSV2(
            opt.video_path,
            opt.something_train_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path)
    elif opt.dataset == 'something-something-v1':
        training_data = SSV1(
            opt.video_path,
            opt.something_train_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            labels_path=opt.something_label_path)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    global validation_data
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'something-something-v2',
                           'something-something-v1']

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
    elif opt.dataset == 'something-something-v2':
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
    elif opt.dataset == 'something-something-v1':
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
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    global subset, test_data
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'something-something-v2',
                           'something-something-v1']
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
    elif opt.dataset == 'something-something-v2':
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
    elif opt.dataset == 'something-somehting-v1':
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

    return test_data
