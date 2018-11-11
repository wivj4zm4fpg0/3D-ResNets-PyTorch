import pandas as pd

import datasets.base


class Shoplifting(datasets.base.BaseLoader):
    def load_annotation_data(self, data_file_path):
        return pd.read_csv(data_file_path, sep=' ')

    def get_class_labels(self, data):
        return {0: 0, 1: 1}

    def get_video_names_and_annotations(self, data, subset):
        video_names = []
        annotations = []

        for i in range(len(data)):
            this_subset = data['subset'][i]
            if this_subset == subset:
                label = data['class'][i]
                if data['class'][i] == 0:
                    dir_name = 'no_action'
                else:
                    dir_name = 'action'
                video_names.append('{}/{}'.format(dir_name, data['video_name'][i]))
                annotations.append({'label': label})
        return video_names, annotations

    def __init__(
            self,
            paths,
            annotation_path,
            subset,
            n_samples_for_each_video=1,
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None,
            sample_duration=32,
            image_format='image_{0:05d}.jpg'
    ):
        super().__init__(
            paths,
            annotation_path,
            subset,
            n_samples_for_each_video,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration,
            image_format,
        )
