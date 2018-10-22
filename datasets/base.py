import copy
import math
import os
from abc import ABCMeta, abstractmethod

import torch
import torch.utils.data as data
from PIL import Image

from utils import load_value_file


def images_loader(paths):
    images = []
    for path in paths:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if paths[path] == '3ch':
                    images.append(img.convert('RGB'))
                elif paths[path] == '1ch':
                    images.append(img.convert('L'))
    return images


def video_loader(video_dir_paths, frame_indices, image_format):
    video = []
    for i in frame_indices:
        images_paths = {}
        for video_dir_path in video_dir_paths:
            path = os.path.join(video_dir_path, image_format.format(i))
            assert os.path.exists(path), 'No such file :{}'.format(path)
            images_paths[path] = video_dir_paths[video_dir_path]
        images = images_loader(images_paths)
        video.extend(images)
    return video


# 複数の画像をチャンネル結合して一つの画像にする
# example :RGB, Gray, Gray, RGB, Gray, Gray, RGB, Gray... -> 5ch, 5ch...
def channels_coupling(clip, n_image):
    new_clip = []
    for i in range(0, len(clip), n_image):
        temp = []
        for j in range(n_image):
            temp.append(clip[i + j])
        new_clip.append(torch.cat(temp, 0))
    return new_clip


class BaseLoader(data.Dataset, metaclass=ABCMeta):

    @abstractmethod
    def load_annotation_data(self, data_file_path):
        pass

    @abstractmethod
    def get_class_labels(self, entry):
        pass

    @abstractmethod
    def get_video_names_and_annotations(self, entry, subset):
        pass

    def make_data_set(self, paths, annotation_path, subset, n_samples_for_each_video,
                      sample_duration):
        entry = self.load_annotation_data(annotation_path)
        video_names, annotations = self.get_video_names_and_annotations(entry, subset)
        class_to_idx = self.get_class_labels(entry)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name
        data_set = []
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('data_set loading [{}/{}]'.format(i, len(video_names)))
            full_paths = {}
            n_frames = 0
            for path in paths:
                full_path = os.path.join(path, video_names[i])
                assert os.path.exists(full_path), 'No such file :{}'.format(full_path)
                full_paths[full_path] = paths[path]
                n_frames_file_path = os.path.join(full_path, 'n_frames')
                if os.path.exists(n_frames_file_path):
                    n_frames = int(load_value_file(n_frames_file_path))
            assert n_frames > 0

            begin_t = 1
            end_t = n_frames
            sample = {
                'paths': full_paths,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': video_names[i].split('/')[1]
            }
            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotations[i]['label']]
            else:
                sample['label'] = -1

            if n_samples_for_each_video == 1:
                sample['frame_indices'] = list(range(1, n_frames + 1))
                data_set.append(sample)
            else:
                if n_samples_for_each_video > 1:
                    step = max(1, math.ceil((n_frames - 1 - sample_duration) / (n_samples_for_each_video - 1)))
                else:
                    step = sample_duration
                for path in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(path, min(n_frames + 1, path + sample_duration)))
                    data_set.append(sample_j)
        return data_set, idx_to_class

    def __init__(self,
                 paths,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=None,
                 image_format='image_{0:05d}.jpg'):
        if get_loader is None:
            get_loader = video_loader
        self.data, self.class_names = self.make_data_set(
            paths, annotation_path, subset, n_samples_for_each_video, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader

        self.image_format = image_format
        self.n_image = len(paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        paths = self.data[index]['paths']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(paths, frame_indices, self.image_format)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.n_image > 1:
            clip = channels_coupling(clip, self.n_image)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
