class ClassLabel(object):

    def __init__(self, flag: bool = False):
        self.flag = flag

    def __call__(self, target: dict):
        if self.flag:
            return target['label'], target['video_id']
        else:
            return target['label']
