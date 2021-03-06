from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# General experiments options
__C.DATASET_NAME = 'MNIST'
__C.GPU_ID = 0
__C.LATENT_SPACE_SIZE = 512

# Training options of vae for MNIST
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_COPY = 4
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 5000
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.OUTPUT_HEIGHT = 28
__C.TRAIN.OUTPUT_WIDTH = 28
__C.TRAIN.INPUT_HEIGHT = 28
__C.TRAIN.INPUT_WIDTH = 28
__C.TRAIN.C_DIM = 1
__C.TRAIN.VAE_LEARNING_RATE = 1e-4
__C.TRAIN.VAE_PRETRAINED_MODEL = ''
__C.TRAIN.LR_DECAY_EPOCH = 50

# Logs Directory
__C.TRAIN.MNIST_VAE_DIR = '/home/exx/Rudra/variational-autoencoder/model_output/vae/MNIST/'
__C.TRAIN.VAEMNIST_LOG_DIR = '//home/exx/Rudra/variational-autoencoder/ckt_logs/ckt_logs_vae_model/mnist/'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_edict(yaml_cfg):
    _merge_a_into_b(yaml_cfg, __C)
