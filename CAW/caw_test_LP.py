"""
Test a trained CAW model with an arbitrary set of edges; the model is assumed to be trained previously.

Date: 
    - Jan. 04, 2023
"""

import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
from data_loading import *
from ne_rand_sampler import *
from test_data_generate import *

#import numba
from module import CAWN
from graph import NeighborFinder
import resource

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
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
NEG_SAMPLE = args.neg_sample
NEG_SAMPLE_TEST = args.neg_sample_test
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio
TS_RND_NE_RATIO_TEST = args.ts_rnd_ne_ratio_test

EGO_SNAP_TEST = False

assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, get_best_model_path = set_up_logger(args, sys_argv)


# ==================== load the data
node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger)

max_idx = max(full_data.sources.max(), full_data.destinations.max())
assert(np.unique(np.stack([full_data.sources, full_data.destinations])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))  # all nodes except node 0 should appear and be compactly indexed
assert(node_features.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))  # the nodes need to map one-to-one to the node feat matrix

# ====================
# create two neighbor finders to handle graph extraction.
# the train and validation use partial ones, while test phase always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(full_data.sources, full_data.destinations, full_data.edge_idxs, full_data.timestamps):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_data.sources, train_data.destinations, train_data.edge_idxs, train_data.timestamps):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_data.sources, val_data.destinations, val_data.edge_idxs, val_data.timestamps):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder

# ====================
# create random samplers to generate train/val/test instances
test_rand_sampler = RandEdgeSampler_NE(full_data.sources, full_data.destinations, full_data.timestamps,
                                       val_data.timestamps[-1], NS=NEG_SAMPLE_TEST, seed=2, rnd_sample_ratio=TS_RND_NE_RATIO_TEST)
nn_test_rand_sampler = RandEdgeSampler_NE(new_node_test_data.sources, new_node_test_data.destinations,
                                          new_node_test_data.timestamps, val_data.timestamps[-1], NS=NEG_SAMPLE_TEST,
                                          seed=3, rnd_sample_ratio=TS_RND_NE_RATIO_TEST)

# ====================
# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# ====================
# device initialization
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# ====================
# the main flow...
for i_run in range(args.n_runs):
    # Filename & Path initialization
    ts_stats_partial_path = f"./pred_stats/{DATA}/"
    ts_stats_partial_path_name = f"CAW-{DATA}-TR_{NEG_SAMPLE}-TS_{NEG_SAMPLE_TEST}-TR_{TR_RND_NE_RATIO}-TS_{TS_RND_NE_RATIO}-TEST_{TS_RND_NE_RATIO_TEST}-{i_run}"
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
    cawn = CAWN(node_features, edge_features, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
    # load the saved model
    cawn.load_state_dict(torch.load(get_best_model_path(i_run)))
    cawn = cawn.to(device)
    cawn.eval()

    # ==================== TEST
    cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
    # ===== Transductive 
    # original setting
    test_perf_dict = eval_link_pred(cawn, test_rand_sampler, test_data, stats_filename=test_stats_path_trans, batch_size=BATCH_SIZE)
    for metric_name, metric_value in test_perf_dict.items():
        logger.info('Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    
    # ego-snapshot-based setting
    if EGO_SNAP_TEST:
        test_pos_snapshots = generate_pos_graph_snapshots(test_data, NUM_SNAPSHOTS[DATA])
        for ts_snp_idx, pos_snapshot in enumerate(test_pos_snapshots):
            logger.info(f"INFO: Test Transductive: Evaluation of the snapshot {ts_snp_idx} started.")
            snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
            save_log_filename = f'{test_stats_path_SNAPSHOT_trans}_{ts_snp_idx}'
            eval_link_pred_one_snapshot(cawn, snapshot_dict, stats_filename=save_log_filename)

    # ===== Inductive
    # orginal setting
    nn_test_perf_dict = eval_link_pred(cawn, nn_test_rand_sampler, new_node_test_data, stats_filename=test_stats_path_induc, batch_size=BATCH_SIZE)
    for metric_name, metric_value in nn_test_perf_dict.items():
        logger.info('Test statistics: New nodes -- {}: {}'.format(metric_name, metric_value))
    
    # ego-snapshot-based setting
    if EGO_SNAP_TEST:
        nn_test_pos_snapshots = generate_pos_graph_snapshots(new_node_test_data, NUM_SNAPSHOTS[DATA])
        for ts_snp_idx, pos_snapshot in enumerate(nn_test_pos_snapshots):
            logger.info(f"INFO: Test Inductive: Evaluation of the snapshot {ts_snp_idx} started.")
            snapshot_dict = generate_test_edge_for_one_snapshot(pos_snapshot, full_data)
            save_log_filename = f'{test_stats_path_SNAPSHOT_induc}_{ts_snp_idx}'
            eval_link_pred_one_snapshot(cawn, snapshot_dict, stats_filename=save_log_filename)


    logger.info('TEST: Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))

