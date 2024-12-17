import numpy as np
from functools import partial
import torch
from lib.timer import Timer
from lib.utils import load_obj, natural_key
from datasets.indoor import IndoorDataset
from datasets.modelnet import get_train_datasets, get_test_datasets
import os,re,sys,json,yaml,random, argparse, torch, pickle
from easydict import EasyDict as edict
from configs.models import architectures
from models.architectures import KPFCNN

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    
    config = dict()
    for key, value in cfg.items():
        for k,v in value.items():
            config[k] = v

    return config

  # load configs
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/train/indoor.yaml', help= 'Path to the config file.')
args = parser.parse_args()
config = load_config(args.config)
config = edict(config)

if torch.cuda.is_available():
        config.device = torch.device('cuda')
else:
        config.device = torch.device('cpu')
    

    # model initialization
config.architecture = architectures[config.dataset]
config.model = KPFCNN(config)



def get_datasets(config):
    info_train = load_obj(config.train_info)
    print(info_train)
    train_set = IndoorDataset(info_train,config,data_augmentation=True)
    return train_set



if __name__ == '__main__':
    train_set = get_datasets(config)
    print(train_set[0])


    
    src1_inds2d, src1_inds3d = projection.projection(src_init_pcd, src_depth_image1, src1_world2camera)