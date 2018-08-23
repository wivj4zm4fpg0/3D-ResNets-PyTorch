import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    images = []
    for i in range(3):
        with open(path[i], 'rb') as f:
            with Image.open(f) as img:
                if i == 0:
                    images.append(img.convert('RGB'))
                else:
                    images.append(img.convert('L'))
    return images


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path[0], '_{:05d}.jpg'.format(i))
        flow_x_path = os.path.join(video_dir_path[1], '_{:05d}.jpg'.format(i))
        flow_y_path = os.path.join(video_dir_path[2], '_{:05d}.jpg'.format(i))
        assert os.path.exists(image_path) and os.path.exists(flow_x_path) and os.path.exists(flow_y_path)
        if os.path.exists(image_path):
            images = image_loader([image_path, flow_x_path, flow_y_path])
            video.append(images[0])
            video.append(images[1])
            video.append(images[2])
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(labels_path):
    data = load_annotation_data(labels_path)
    class_labels_map = {}
    for entry in data.items():
        class_labels_map[entry[0]] = int(entry[1])
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for entry in data:
        video_names.append(entry['id'])
        if not subset == 'test':
            label = {'label': entry['template']}
            annotations.append(label)

    return video_names, annotations


def make_dataset(rgb_images_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, labels_path, flow_x_images_path, flow_y_images_path):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(labels_path)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(rgb_images_path, video_names[i])  # root_pathとvideo_names[i]を連結してビデオファイルのパスを完成させる
        flow_x_path = os.path.join(flow_x_images_path, video_names[i])
        flow_y_path = os.path.join(flow_y_images_path, video_names[i])
        assert os.path.exists(video_path) and os.path.exists(flow_x_path) and os.path.exists(
            flow_y_path)  # ファイルが存在するか確かめる

        n_frames_file_path = os.path.join(video_path, 'n_frames')  # 動画のフレーム数が記録されているn_framesのパスを完成させる
        assert os.path.exists(n_frames_file_path)  # ファイルが存在するか確かめる
        n_frames = int(load_value_file(n_frames_file_path))  # n_framesを読み込みフレーム数を入力する
        if n_frames <= 0:
            continue

        begin_t = 1  # フレームの始まり
        end_t = n_frames  # フレームの終わり
        sample = {  # 動画のメタ情報を持つ辞書変数
            'video': video_path,
            'flow_x': flow_x_path,
            'flow_y': flow_y_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i]
        }
        assert len(annotations) != 0
        sample['label'] = class_to_idx[annotations[i]['label']]

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class SSV2FLOW(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 rgb_images_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 labels_path="",
                 flow_x_images_path="",
                 flow_y_images_path=""):
        self.data, self.class_names = make_dataset(
            rgb_images_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, labels_path=labels_path, flow_x_images_path=flow_x_images_path,
            flow_y_images_path=flow_y_images_path)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = [self.data[index]['video'], self.data[index]['flow_x'], self.data[index]['flow_y']]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = concate_channels(clip)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)

def concate_channels(tensors):
    clip = []
    for i in range(0, len(tensors), 3):
        clip.append(torch.cat([tensors[i], tensors[i + 1], tensors[i + 2]], 0))
    return clip