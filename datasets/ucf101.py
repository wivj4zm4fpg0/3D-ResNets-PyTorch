import datasets.base
import json
from torchvision import transforms

from target_transforms import ClassLabel


class UCF101(datasets.base.BaseLoader):
    def load_annotation_data(self, data_file_path: str) -> dict:
        with open(data_file_path, 'r') as data_file:
            return json.load(data_file)

    def get_class_labels(self, data: dict) -> dict:
        class_labels_map = {}
        index = 0
        for class_label in data['labels']:
            class_labels_map[class_label] = index
            index += 1
        return class_labels_map

    def get_video_names_and_annotations(self, data: dict, subset: str) -> tuple:
        video_names = []
        annotations = []

        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])
        return video_names, annotations

    def __init__(
            self,
            paths: dict,
            annotation_path: str,
            subset: str,
            spatial_transform: transforms = None,
            target_transform: ClassLabel = None,
            image_format='image_{0:05d}.jpg'
    ):
        super().__init__(
            paths,
            annotation_path,
            subset,
            spatial_transform,
            target_transform,
            image_format
        )
