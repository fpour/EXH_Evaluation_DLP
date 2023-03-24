"""
Test a trained TGN model with a set of edges of interest

Date:
    - March 09, 2023

"""

import math
import logging
import time
import sys
import torch
import numpy as np
import pandas as np
import pickle
from pathlib import Path

from evaluation.evaluation_LP import eval_link_pred, eval_link_pred_one_snapshot, eval_link_pred_for_HitsK
from model.tgn import TGN
from utils.utils import *
from utils.data_load import get_data
from utils.neg_edge_sampler import NegativeEdgeSampler
from utils.log import *
from train.train_LP import train_val_LP
from utils.test_data_generate import generate_test_edge_for_one_snapshot, generate_pos_graph_snapshots
from utils.arg_parser import get_args

# Hyper-parameters based on the characteristics of the datasets
NUM_SNAPSHOTS = {'canVote': 2,
                 'LegisEdgelist': 1,
                 'enron': 10,
                 'mooc': 10,
                 'reddit': 10,
                 'uci': 10,
                 'wikipedia': 10,
                 }

# parse the arguments
args, sys_argv = get_args()

# set parameters
DATA = args.data
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
NUM_NEIGHBORS = args.n_degree
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
NUM_LAYER = args.n_layer
GPU = args.gpu
BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
LEARNING_RATE = args.lr
BACKPROP_EVERY = args.backprop_every
SEED = args.seed
EGO_SNAP = args.ego_snap
TR_NEG_SAMPLE = args.tr_neg_sample
TS_NEG_SAMPLE = args.ts_neg_sample
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio

# for saving the results...
meta_info = {'model': args.prefix,
            'data': DATA,
            'tr_neg_sample': TR_NEG_SAMPLE,
            'ts_neg_sample': TS_NEG_SAMPLE,
            'tr_rnd_ne_ratio': TR_RND_NE_RATIO,
            'ts_rnd_ne_ratio': TS_RND_NE_RATIO,
            }

# ===================================== Set the seed, logger, and file paths
set_random_seed(SEED)
logger, get_checkpoint_path, get_best_model_path = set_up_log_path(args, sys_argv)

# ===================================== load the data
node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger)

# ===================================== Create neighbor samplers
# create two neighbor finders to handle graph extraction.
# the train and validation use partial ones, while test phase always uses the full one
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
partial_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# ===================================== Negative Edge Samplers
# create random samplers to generate train/val/test instances for the negative edges
last_ts_before_test = val_data.timestamps[-1]
# train
# train_rand_sampler = NegativeEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps,
#                                         last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO)
# # validation
# val_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
#                                        last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO, seed=0)
# nn_val_rand_sampler = NegativeEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, new_node_val_data.timestamps,
#                                           last_ts_before_test, NS=TR_NEG_SAMPLE, rnd_sample_ratio=TR_RND_NE_RATIO, seed=1)

# test
test_rand_sampler = NegativeEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps,
                                        last_ts_before_test, NS=TS_NEG_SAMPLE, seed=2, rnd_sample_ratio=TS_RND_NE_RATIO)
nn_test_rand_sampler = NegativeEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, 
                                           last_ts_before_test, NS=TS_NEG_SAMPLE,
                                           seed=3, rnd_sample_ratio=TS_RND_NE_RATIO)

# ===================================== set up the device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# ===================================== The main flow ...
# compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

Path("./LP_stats/").mkdir(parents=True, exist_ok=True)
Path(f"./LP_stats/{DATA}/").mkdir(parents=True, exist_ok=True)
Path(f"./LP_stats/{DATA}/snapshots").mkdir(parents=True, exist_ok=True)

for i_run in range(args.start_run_idx, args.n_runs):

    start_time_run = time.time()
    logger.info("="*50)
    logger.info("********** Run {} starts. **********".format(i_run))

    # stats file name initialization
    ts_stats_partial_path = f"LP_stats/{DATA}/"
    ts_stats_partial_name = f"{args.prefix}_{DATA}_TR_{TR_NEG_SAMPLE}_TS_{TS_NEG_SAMPLE}_{i_run}"
    # # original standard evaluation statistis
    # ts_stats_path_trans = f'{ts_stats_partial_path}/{ts_stats_partial_name}_trans'  # Old nodes
    # ts_stats_path_induc = f'{ts_stats_partial_path}/{ts_stats_partial_name}_induc'  # New nodes
    # ts_STD_pred_trans = "LP_stats/STD_pred_TRANS.csv"
    # ts_STD_pred_induc = "LP_stats/STD_pred_INDUC.csv"
    # # snapshot-based evaluation statistics
    # if EGO_SNAP:
    #     ts_stats_path_trans_SNAPSHOT = f'{ts_stats_partial_path}/snapshots/{ts_stats_partial_name}_trans'
    #     ts_stats_path_induc_SNAPTHOT = f'{ts_stats_partial_path}/snapshots/{ts_stats_partial_name}_induc'

    # original standard evaluation statistis
    ts_stats_path_trans = f'{ts_stats_partial_path}/hitsK/{ts_stats_partial_name}_trans'  # Old nodes
    ts_stats_path_induc = f'{ts_stats_partial_path}/hitsK/{ts_stats_partial_name}_induc'  # New nodes

    # model initialization
    tgn = TGN(neighbor_finder=partial_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep, decoder='LP')
    tgn.load_state_dict(torch.load(get_best_model_path(i_run)))
    tgn = tgn.to(device)
    tgn.eval()

    # ===================================== TEST
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    # ========= Transductive
    # logger.info("TEST: Standard Setting: Transductive")
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # test_perf_dict = eval_link_pred(model=tgn, sampler=test_rand_sampler, data=test_data, logger=logger, stats_filename=ts_stats_path_trans, 
    #                                 batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    # for metric_name, metric_value in test_perf_dict.items():
    #     logger.info('INFO: Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    # # write the summary of prediction results to a csv file
    # dict_list = [meta_info, test_perf_dict]
    # write_dicts_to_csv(dict_list, ts_STD_pred_trans)

    # if EGO_SNAP:
    #     logger.info("TEST: EGO-SNAPSHOT Setting: Transductive")
    #     if USE_MEMORY:
    #         tgn.memory.restore_memory(val_memory_backup)
        
    #     test_pos_snapshots = generate_pos_graph_snapshots(test_data, NUM_SNAPSHOTS[DATA])
    #     for ts_snp_idx, pos_snapshot in enumerate(test_pos_snapshots):
    #             logger.info(f"INFO: Transductive: Evaluation of the snapshot {ts_snp_idx} started.")
    #             snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
    #             save_log_filename = f'{ts_stats_path_trans_SNAPSHOT}_{ts_snp_idx}'
    #             eval_link_pred_one_snapshot(model=tgn, snap_data=snapshot_dict, logger=logger, stats_filename=save_log_filename, 
    #                                         batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)

    # Hits@K evaluation setup:
    logger.info("TEST: Hits@k Evaluation Setup: Transductive")
    hits_against_size = 100
    eval_link_pred_for_HitsK(model=tgn, sampler=test_rand_sampler, data=test_data, logger=logger, size=hits_against_size, 
                             stats_filename=ts_stats_path_trans, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)

    # ========= Inductive
    # logger.info("TEST: Standard Setting: Inductive")
    if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)

    # nn_test_measures_dict = eval_link_pred(model=tgn, sampler=nn_test_rand_sampler, data=new_node_test_data, logger=logger,
    #                                         stats_filename=ts_stats_path_induc, 
    #                                         batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    # for measure_name, measure_value in nn_test_measures_dict.items():
    #         logger.info('INFO: Test statistics: New nodes -- {}: {}'.format(measure_name, measure_value))
    # # write the summary of prediction results to a csv file
    # dict_list = [meta_info, nn_test_measures_dict]
    # write_dicts_to_csv(dict_list, ts_STD_pred_induc)
    
    # if EGO_SNAP:
    #     logger.info("TEST: EGO-SNAPSHOT Setting: Inductive")
    #     if USE_MEMORY:
    #         tgn.memory.restore_memory(val_memory_backup)
        
    #     nn_test_pos_snapshots = generate_pos_graph_snapshots(new_node_test_data, NUM_SNAPSHOTS[DATA])
    #     for nn_ts_snp_idx, pos_snapshot in enumerate(nn_test_pos_snapshots):
    #         logger.info(f"INFO: Inductive: Evaluation of the snapshot {nn_ts_snp_idx} started.")
    #         snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
    #         save_log_filename = f'{ts_stats_path_induc_SNAPTHOT}_{nn_ts_snp_idx}'
    #         eval_link_pred_one_snapshot(model=tgn, snap_data=snapshot_dict, logger=logger, stats_filename=save_log_filename,
    #                                     batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    
    logger.info("TEST: Hits@k Evaluation Setup: Inductive")
    hits_against_size = 100
    eval_link_pred_for_HitsK(model=tgn, sampler=nn_test_rand_sampler, data=new_node_test_data, logger=logger, size=hits_against_size, 
                             stats_filename=ts_stats_path_induc, batch_size=BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    
    
   
   # ===================================== 

    logger.info('TEST: Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))
    logger.info("="*50)

