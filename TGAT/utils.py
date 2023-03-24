import numpy as np
import os
import argparse
import sys
import torch
import random

def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
    # select data
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--data_usage', default=1.0, type=float, help='fraction of data to use (0-1)')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for validation and test.')
    parser.add_argument('--prefix', type=str, default='TGAT', help='prefix to name the checkpoints')
    # method-related hyper-parameters
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    # general training hyper-parameters
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')
    # parameters controlling the computation setting sbut not affecting the resutls in general
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    # parameters controlling the valisation and test set
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio of the test data.")
    parser.add_argument('--tr_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during training.')
    parser.add_argument('--ts_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during TEST phase.')
    parser.add_argument('--ts_rnd_ne_ratio_test', type=float, default=0.0,
                        help='Ratio of random negative edges sampled during TEST phase.')
    parser.add_argument('--neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'hist', 'induc', 'haphaz_rnd'],
                        help='Strategy for the edge negative sampling.')
    parser.add_argument('--neg_sample_test', type=str, default='hist', choices=['rnd', 'hist', 'induc', 'haphaz_rnd'],
                        help='Strategy for the edge negative sampling ONLY for testing.')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    return args, sys.argv


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

