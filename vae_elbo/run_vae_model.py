from __future__ import division
from __future__ import print_function

import datetime
import argparse
import pprint
import time

from vae_elbo.model_mnist import VAEMnist
from vae_elbo.trainer_mnist import VAEMnsitTrainer
from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE network")
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print("Using config:")
    pprint.pprint(cfg)

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME

    model = VAEMnist(image_shape=[28, 28, 1])
    algo = VAEMnsitTrainer(model=model)

    algo.train()
