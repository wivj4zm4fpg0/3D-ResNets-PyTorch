from abc import ABCMeta, abstractclassmethod
from PIL import Image
import os
import torch.utils.data as data
import torch
import math
import copy

from utils import load_value_file


class BaseLoader(data.Dataset, metaclass=ABCMeta):
    def images_loader(self, paths):
        images = []
        for i in range(len(paths)):
            with open(paths[i], 'rb') as f:
                with Image.open(f) as img:
                    if i == 0:
                        images.append(img.convert('RGB'))
                    else:
                        images.append(img.convert('L'))
        return images

    def video_loader(self, video_dir_paths, frame_indices, image_format):
        video = []
        for i in frame_indices:
            images_paths = []
            for j in range(len(video_dir_paths)):
                path = os.path.join(video_dir_paths[j], image_format.format(i))
                assert os.path.exists(path), 'No such file :{}'.format(path)
                images_paths.append(path)
            images = self.images_loader(images_paths)
            for j in range(len(video_dir_paths)):
                video.append(images[j])
        return video

    @abstractclassmethod
    def load_annotation_data(self, data_file_path):
        pass

    @abstractclassmethod
    def get_class_labels(self, data):
        pass

    @abstractclassmethod
    def get_video_names_and_annotations(self, data, subset):
        pass

    def make_dataset(self, paths, annotation_path, subset, n_samples_for_each_video,
                     sample_duration):
        data = self.load_annotation_data(annotation_path)
        video_names, annotations = self.get_video_names_and_annotations(data, subset)
        class_to_idx = self.get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name
        dataset = []
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_names)))
            full_paths = []
            for j in range(len(paths)):
                full_path = os.path.join(paths[j], video_names[i])
                assert os.path.exists(full_path), 'No such file :{}'.format(full_path)
                full_paths.append(full_path)
            n_frames_file_path = os.path.join(full_paths[0], 'n_frames')
            assert os.path.exists(n_frames_file_path)
            n_frames = int(load_value_file(n_frames_file_path))

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

    def concate_channels(self, clip, n_channel):
        new_clip = []
        for i in range(0, len(clip), n_channel):
            temp = []
            for j in range(n_channel):
                temp.append(clip[i + j])
            new_clip.append(torch.cat(temp, 0))
        return new_clip

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
                 image_format='_{0:05d}.jpg',
                 n_channel=3):
        if get_loader is None:
            get_loader = self.video_loader
        self.data, self.class_names = self.make_dataset(
            paths, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader

        self.image_format = image_format
        self.n_channel = n_channel

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['paths']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.image_format)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.n_channel > 3:
            clip = self.concate_channels(clip, self.n_channel - 2)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
