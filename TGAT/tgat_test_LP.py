"""
Test a trained TGAT model with an arbitrary set of edges; the model is assumed to be trained previously.

Date: 
    - Jan. 20, 2023
"""

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
from ne_rand_sampler import RandEdgeSampler, NegativeEdgeSampler
from train import train_val
from test_data_generate import generate_pos_graph_snapshots, generate_test_edge_for_one_snapshot


NUM_SNAPSHOTS = {'canVote': 2,
                 'LegisEdgelist': 1,
                 'enron': 10,
                 'mooc': 10,
                 'reddit': 10,
                 'uci': 10,
                 'wikipedia': 10,
                 }


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
NEG_SAMPLE_TEST = args.neg_sample_test
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio
TS_RND_NE_RATIO_TEST = args.ts_rnd_ne_ratio_test

INIT_FEAT_DIM = 172
NO_INIT_FEAT_DATA_LIST = ['enron', 'mooc']  # ['uci', 'enron']

EGO_SNAP_TEST = False

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
test_rand_sampler = RandEdgeSampler_NE(full_data.sources, full_data.destinations, full_data.timestamps,
                                       val_data.timestamps[-1], NS=NEG_SAMPLE_TEST, seed=2, rnd_sample_ratio=TS_RND_NE_RATIO_TEST)
nn_test_rand_sampler = RandEdgeSampler_NE(new_node_test_data.sources, new_node_test_data.destinations,
                                          new_node_test_data.timestamps, val_data.timestamps[-1], NS=NEG_SAMPLE_TEST,
                                          seed=3, rnd_sample_ratio=TS_RND_NE_RATIO_TEST)

# ====================
# device initialization
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)


# ==================== The Main Flow ...
for i_run in range(args.n_runs):
    # Filename & Path initialization
    ts_stats_partial_path = f"./pred_stats/{DATA}/"
    ts_stats_partial_path_name = f"TGAT-{DATA}-TR_{NEG_SAMPLE}-TS_{NEG_SAMPLE_TEST}-TR_{TR_RND_NE_RATIO}-TS_{TS_RND_NE_RATIO}-TEST_{TS_RND_NE_RATIO_TEST}-{i_run}"
    # test_stats_path_ALL = f'{ts_stats_partial_path}/{ts_stats_partial_path_name}_TEST_ALL'
    test_stats_path_trans = f'{ts_stats_partial_path}/{ts_stats_partial_path_name}_TEST_Ntrans'
    test_stats_path_induc = f'{ts_stats_partial_path}/{ts_stats_partial_path_name}_TEST_Ninduc'
    
    if EGO_SNAP_TEST:
        test_stats_path_SNAPSHOT_trans = f'{ts_stats_partial_path}/snapshots/{ts_stats_partial_path_name}_TEST_Ntrans'
        test_stats_path_SNAPSHOT_induc = f'{ts_stats_partial_path}/snapshots/{ts_stats_partial_path_name}_TEST_Ninduc'


    start_time_run = time.time()
    logger.info("*"*50)
    logger.info("********** TEST: Run {} starts. **********".format(i_run))

    # model initialization
    tgat = TGAN(partial_ngh_finder, node_features, edge_features,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM, 
            get_checkpoint_path=get_checkpoint_path)
    # load the saved model 
    tgat.load_state_dict(torch.load(get_best_model_path(i_run)))
    tgat = tgat.to(device)
    tgat.eval()

    # ==================== TEST
    tgat.ngh_finder = full_ngh_finder
    # ===== Transductive 
    # original setting
    test_perf_dict = eval_link_pred(tgat, test_rand_sampler, test_data, stats_filename=test_stats_path_trans, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for metric_name, metric_value in test_perf_dict.items():
        logger.info('Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    
    # ego-snapshot-based setting
    if EGO_SNAP_TEST:
        test_pos_snapshots = generate_pos_graph_snapshots(test_data, NUM_SNAPSHOTS[DATA])
        for ts_snp_idx, pos_snapshot in enumerate(test_pos_snapshots):
            logger.info(f"INFO: Test Transductive: Evaluation of the snapshot {ts_snp_idx} started.")
            snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
            save_log_filename = f'{test_stats_path_SNAPSHOT_trans}_{ts_snp_idx}'
            eval_link_pred_one_snapshot(tgat, snapshot_dict, stats_filename=save_log_filename)

    # ===== Inductive
    # orginal setting
    nn_test_perf_dict = eval_link_pred(tgat, nn_test_rand_sampler, new_node_test_data, stats_filename=test_stats_path_induc, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for metric_name, metric_value in nn_test_perf_dict.items():
        logger.info('Test statistics: New nodes -- {}: {}'.format(metric_name, metric_value))
    
    # ego-snapshot-based setting
    if EGO_SNAP_TEST:
        nn_test_pos_snapshots = generate_pos_graph_snapshots(new_node_test_data, NUM_SNAPSHOTS[DATA])
        for ts_snp_idx, pos_snapshot in enumerate(nn_test_pos_snapshots):
            logger.info(f"INFO: Test Inductive: Evaluation of the snapshot {ts_snp_idx} started.")
            snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
            save_log_filename = f'{test_stats_path_SNAPSHOT_induc}_{ts_snp_idx}'
            eval_link_pred_one_snapshot(tgat, snapshot_dict, stats_filename=save_log_filename)


    logger.info('TEST: Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))



 




