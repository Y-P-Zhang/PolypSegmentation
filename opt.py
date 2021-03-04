import argparse
import os
import yaml
from utils.loss import criterion

parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')

parse.add_argument('--config',
                   help='Path to the training config file.', required=True)
parse.add_argument('--start_epoch', type=int, default=1)
parse.add_argument('--continue_train', action='store_true')
parse.add_argument('--gpu_id', type=str, default="0")
parse.add_argument('--mode', type=str, default="train")
parse.add_argument('--exp_name', type=str, default="")


class Config(dict):
    def __init__(self, filename):
        super().__init__()
        with open('./configs/' + filename + '.yaml', 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)

        for k, v in cfg_dict.items():
            self.__dict__[k] = v
        self.criterion = criterion(cfg_dict['loss'])


option = parse.parse_args()
opt = Config(option.config)

if option.exp_name == "":
    opt.exp_name = opt.model + '_' + str(option.config)
else:
    opt.exp_name = option.exp_name
print('exp_name:', opt.exp_name)

opt.mode = option.mode
opt.start_epoch = option.start_epoch
opt.continue_train = option.continue_train

os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id
