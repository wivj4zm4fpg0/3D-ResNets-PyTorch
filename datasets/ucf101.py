import datasets.base
import json


class UCF101(datasets.base.BaseLoader):
    def load_annotation_data(self, data_file_path):
        with open(data_file_path, 'r') as data_file:
            return json.load(data_file)

    def get_class_labels(self, data):
        class_labels_map = {}
        index = 0
        for class_label in data['labels']:
            class_labels_map[class_label] = index
            index += 1
        return class_labels_map

    def get_video_names_and_annotations(self, data, subset):
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
            paths,
            annotation_path,
            subset,
            spatial_transform=None,
            target_transform=None,
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
