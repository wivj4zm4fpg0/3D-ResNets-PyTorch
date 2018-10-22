from datasets.activitynet import ActivityNet
from datasets.hmdb51 import HMDB51
from datasets.kinetics import Kinetics
from datasets.ssv1 import SSV1
from datasets.ssv2 import SSV2
from datasets.ucf101 import UCF101
from datasets.shoplifting import Shoplifting

datasets = {'ucf101': UCF101, 'kinetics': Kinetics, 'activitynet': ActivityNet,
            'hmdb51': HMDB51, 'ssv1': SSV1, 'ssv2': SSV2, 'shoplifting': Shoplifting}
