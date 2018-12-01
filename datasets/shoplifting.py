import pandas as pd
from pandas.core.frame import DataFrame
from torchvision import transforms

import datasets.base
from target_transforms import ClassLabel


class Shoplifting(datasets.base.BaseLoader):
    def load_annotation_data(self, data_file_path: str) -> DataFrame:
        return pd.read_csv(data_file_path, sep=' ')

    def get_class_labels(self, data: DataFrame) -> dict:
        return {0: 0, 1: 1}

    def get_video_names_and_annotations(self, data: DataFrame, subset: str) -> tuple:
        video_names = []
        annotations = []

        for i in range(len(data)):
            this_subset = data['subset'][i]
            if this_subset == subset:
                label = data['class'][i]
                if label == 0:
                    dir_name = 'no_action'
                elif label == 1:
                    dir_name = 'action'
                else:
                    raise Exception("invalid value -> data['class'][i]")
                video_names.append('{}/{}'.format(dir_name, data['video_name'][i]))
                annotations.append({'label': label})

        return video_names, annotations

    def __init__(
            self,
            paths: dict,
            annotation_path: str,
            subset: str,
            spatial_transform: transforms = None,
            target_transform: ClassLabel = None,
            image_format: str = 'image_{0:05d}.jpg'
    ):
        super().__init__(
            paths,
            annotation_path,
            subset,
            spatial_transform,
            target_transform,
            image_format,
        )
