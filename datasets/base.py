import os
from abc import ABCMeta, abstractmethod

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from target_transforms import ClassLabel
from utils import load_value_file


def images_loader(paths: dict) -> list:
    images = []
    for path in paths:
        with open(path, 'rb') as f:  # rb->読み込み専用・バイナリ
            with Image.open(f) as img:
                if paths[path] == '3ch':
                    images.append(img.convert('RGB'))
                elif paths[path] == '1ch':
                    images.append(img.convert('L'))
    return images


def video_loader(video_dir_paths: dict, frame_indices: list, image_format: str) -> list:
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
def channels_coupling(clip: list, image_number: int) -> list:
    new_clip = []
    for i in range(0, len(clip), image_number):
        temp = []
        for j in range(image_number):
            temp.append(clip[i + j])
        new_clip.append(torch.cat(temp, 0))
    return new_clip


class BaseLoader(data.Dataset, metaclass=ABCMeta):

    @abstractmethod
    def load_annotation_data(self, data_file_path: str):
        pass

    @abstractmethod
    def get_class_labels(self, entry):
        pass

    @abstractmethod
    def get_video_names_and_annotations(self, entry, subset: str) -> tuple:
        pass

    def make_data_set(
            self,
            paths: dict,
            annotation_path: str,
            subset: str
    ):

        # アノテーションファイルからjsonやcsvオブジェクトを取得する
        entry = self.load_annotation_data(annotation_path)
        # 動画の名前とその属するクラスのそれぞれのリストを取得
        video_names, annotations = self.get_video_names_and_annotations(entry, subset)
        # クラスをidへ割り振る
        class_to_idx = self.get_class_labels(entry)
        # idからクラスがわかるようにする
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        video_information = []

        for i in range(len(video_names)):

            # 1000毎に経過報告をする
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

            video_info = {
                'paths': full_paths,
                'n_frames': n_frames,
                'video_id': video_names[i].split('/')[1],
                'label': class_to_idx[annotations[i]['label']],
                'frame_indices': list(range(1, n_frames + 1))
            }

            video_information.append(video_info)

        return video_information, idx_to_class

    def __init__(
            self,
            paths: dict,
            annotation_path: str,
            subset: str,
            spatial_transform: transforms = None,
            target_transform: ClassLabel = None,
            image_format: str = 'image_{0:05d}.jpg'
    ):
        self.data, self.class_names = self.make_data_set(
            paths,
            annotation_path,
            subset
        )

        self.spatial_transform = spatial_transform
        self.target_transform = target_transform

        self.image_format = image_format
        self.image_number = len(paths)

    def __getitem__(self, index: int) -> tuple:  # イテレートするときに呼び出される
        paths = self.data[index]['paths']
        clip = video_loader(paths, self.data[index]['frame_indices'], self.image_format)
        clip = [self.spatial_transform(img) for img in clip]

        if self.image_number > 1:
            clip = channels_coupling(clip, self.image_number)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            if self.target_transform.flag:
                target, target_name = self.target_transform(target)
                return clip, target, target_name
            else:
                target = self.target_transform(target)

        return clip, target

    def __len__(self) -> int:  # データの総数を定義．どこかで使われるらしい
        return len(self.data)
