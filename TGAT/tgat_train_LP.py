"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import *
from log import set_up_logger
from eval import *
from data_loading import get_data
from ne_rand_sampler import NegativeEdgeSampler
from train import train_val




args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
TOLERANCE = args.tolerance
SEED = args.seed
NEG_SAMPLE = args.neg_sample
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio

INIT_FEAT_DIM = 172
NO_INIT_FEAT_DATA_LIST = ['enron', 'mooc']  # ['uci', 'enron']

# ==================== set the seed, logger, and file paths
set_random_seed(SEED)
logger, get_checkpoint_path, get_best_model_path = set_up_logger(args, sys_argv)


# ==================== load the data
node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger)
max_idx = max(full_data.sources.max(), full_data.destinations.max())

if DATA in NO_INIT_FEAT_DATA_LIST:  # generating arrays of zero elements for the datasets with no edge/node features
    node_zero_padding = np.zeros((node_features.shape[0], INIT_FEAT_DIM - node_features.shape[1]))
    node_features = np.concatenate([node_features, node_zero_padding], axis=1)
    edge_zero_padding = np.zeros((edge_features.shape[0], INIT_FEAT_DIM - edge_features.shape[1]))
    edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)

# ====================
# create two neighbor finders to handle graph extraction.
# the train and validation use partial ones, while test phase always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(full_data.sources, full_data.destinations, full_data.edge_idxs, full_data.timestamps):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_data.sources, train_data.destinations, train_data.edge_idxs, train_data.timestamps):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_data.sources, val_data.destinations, val_data.edge_idxs, val_data.timestamps):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, uniform=UNIFORM)

# ====================
# create random samplers to generate train/val/test instances
train_rand_sampler = NegativeEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = NegativeEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = NegativeEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# ====================
# device initialization
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# ==================== The Main Flow ...
for i_run in range(args.n_runs):

    # stats file-name initialization
    ts_stats_partial_path = f"pred_stats/{DATA}/"
    ts_stats_partial_name = f"TGAT_{DATA}_{NEG_SAMPLE}_{i_run}"
    ts_stats_path_trans = f'{ts_stats_partial_path}/{ts_stats_partial_name}_trans'  # Old nodes
    ts_stats_path_induc = f'{ts_stats_partial_path}/{ts_stats_partial_name}_induc'  # New nodes

    # reset the seed of the train_rand_sampler
    train_rand_sampler.reset_random_state(seed=i_run)

    start_time_run = time.time()
    logger.info("*"*50)
    logger.info("********** Run {} starts. **********".format(i_run))

    # model initialization
    tgat = TGAN(partial_ngh_finder, node_features, edge_features,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM, 
            get_checkpoint_path=get_checkpoint_path)
    tgat = tgat.to(device)
    optimizer = torch.optim.Adam(tgat.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # ==================== TRAINING & VALIDATION 
    # start train and val phases
    train_val_data = (train_data, val_data, new_node_val_data)
    train_val_sampler = (train_rand_sampler, val_rand_sampler, nn_val_rand_sampler)
    train_val(tgat, train_val_data, BATCH_SIZE, NUM_EPOCH, NUM_NEIGHBORS, criterion, optimizer, early_stopper, partial_ngh_finder, 
             train_val_sampler, logger, i_run, device)

    # ==================== TEST
    tgat.ngh_finder = full_ngh_finder
    # ===== Transductive 
    test_perf_dict = eval_link_pred(tgat, test_rand_sampler, test_data, stats_filename=ts_stats_path_trans, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for metric_name, metric_value in test_perf_dict.items():
        logger.info('Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    # ===== Inductive
    nn_test_perf_dict = eval_link_pred(tgat, nn_test_rand_sampler, new_node_test_data, stats_filename=ts_stats_path_induc, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for metric_name, metric_value in nn_test_perf_dict.items():
        logger.info('Test statistics: New nodes -- {}: {}'.format(metric_name, metric_value))

    # ==================== Save model
    logger.info('Run {}: Saving TGAT model ...'.format(i_run))
    torch.save(tgat.state_dict(), get_best_model_path(i_run))
    logger.info('TGAT model saved at {}.'.format(get_best_model_path(i_run)))

    logger.info('Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))



 




