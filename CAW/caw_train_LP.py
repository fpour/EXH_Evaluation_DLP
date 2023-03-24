"""
Modified version of main.py

Date: 
    - Jan. 03, 2023
"""

import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
from data_loading import *
from ne_rand_sampler import *

#import numba
from module import CAWN
from graph import NeighborFinder
import resource

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
TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
TS_RND_NE_RATIO = args.ts_rnd_ne_ratio

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
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

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

    # stats file-name initialization
    ts_stats_partial_path = f"pred_stats/{DATA}/"
    ts_stats_partial_name = f"CAW_{DATA}_{NEG_SAMPLE}_{i_run}"
    ts_stats_path_trans = f'{ts_stats_partial_path}/{ts_stats_partial_name}_trans'  # Old nodes
    ts_stats_path_induc = f'{ts_stats_partial_path}/{ts_stats_partial_name}_induc'  # New nodes


    start_time_run = time.time()
    logger.info("*"*50)
    logger.info("********** Run {} starts. **********".format(i_run))

    # reset the seed of the train_rand_sampler
    train_rand_sampler.reset_random_state(seed=i_run)

    # model initialization
    cawn = CAWN(node_features, edge_features, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
    cawn.to(device)
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # ==================== TRAINING & VALIDATION 
    # start train and val phases
    train_val_data = (train_data, val_data, new_node_val_data)
    train_val_sampler = (train_rand_sampler, val_rand_sampler, nn_val_rand_sampler)
    train_val_1(train_val_data, cawn, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, 
                partial_ngh_finder, train_val_sampler, logger, i_run)

    # ==================== TEST
    cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
    # ===== Transductive 
    test_perf_dict = eval_link_pred(cawn, test_rand_sampler, test_data, stats_filename=ts_stats_path_trans, batch_size=BATCH_SIZE)
    for metric_name, metric_value in test_perf_dict.items():
        logger.info('Test statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))
    # ===== Inductive
    nn_test_perf_dict = eval_link_pred(cawn, nn_test_rand_sampler, new_node_test_data, stats_filename=ts_stats_path_induc, batch_size=BATCH_SIZE)
    for metric_name, metric_value in nn_test_perf_dict.items():
        logger.info('Test statistics: New nodes -- {}: {}'.format(metric_name, metric_value))


    # ==================== Save model
    logger.info('Run {}: Saving CAWN model ...'.format(i_run))
    torch.save(cawn.state_dict(), get_best_model_path(i_run))
    logger.info('CAWN model saved at {}.'.format(get_best_model_path(i_run)))

    logger.info('Run {} elapsed time: {} seconds.'.format(i_run, (time.time() - start_time_run)))

