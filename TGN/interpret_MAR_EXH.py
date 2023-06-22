"""
Interpret the performance of DLP task in EXHaustive mode with MAR (Mean Average Rank) metric
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys

from interpret_res_LP_utils import *
from utils.test_data_generate import get_unique_edges
from utils.data_load import Data, get_data
from utils.utils import set_random_seed
from interpret_res_LP_utils import generate_snapshot_edge_status, gen_meta_info_for_eval



def get_args():
    parser = argparse.ArgumentParser('*** DLP - EXH - MAR ***')
    # Related to stats processing
    parser.add_argument('--prefix', type=str, default='tgn_attn', choices=['tgn_attn', 'jodie_rnn', 'dyrep_rnn', 'CAW', 'TGAT'], help='Model Prefix')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')
    parser.add_argument('--tr_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc'],
                        help='Strategy for the negative sampling at the training phase.')
    parser.add_argument('--ts_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc', 'hitsK'],
                        help='Strategy for the negative edge sampling at the test phase.')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--lp_mode', type=str, default='trans', choices=['trans', 'induc'],
                        help="Link prediction mode: transductive or inductive")
    # Required for the data_loading
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--data_usage', default=1.0, type=float, help='fraction of data to use (0-1)')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for validation and test.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio of the test data.")
    parser.add_argument('--tr_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during TRAINING phase.')
    parser.add_argument('--ts_rnd_ne_ratio', type=float, default=1.0,
                        help='Ratio of random negative edges sampled during TEST phase.')

    try:
        args = parser.parse_args()
        print("INFO: args:", args)
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def invoke_stats_generation(partial_path, partial_filename, stats_filename, full_data, MODEL_NAME, DATA, 
                            LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS):
    """
    Call one instance of the "gen_stats_one_iteration" method
    """
    stats_list = []
    for run_idx in range(N_RUNS):
        for snap_idx in range(NUM_SNAPSHOTS[DATA]):
            iter_partial_filename = f"{partial_path}/{DATA}_snapshots/{partial_filename}_{run_idx}_{LP_MODE}_{snap_idx}"
            if os.path.isfile(f"{iter_partial_filename}_src.npy"):  # if the current snapshot info exists...
                pred_dict = {'sources': np.load(f"{iter_partial_filename}_src.npy"),
                            'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                            'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                            # 'e_idxs': np.load(f"{iter_partial_filename}_e_idx.npy"),
                            'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                            'labels': np.load(f"{iter_partial_filename}_label.npy"),
                            }
                stats_snapshot_dict = gen_stats_one_iteration(pred_dict, full_data)
                stats_snapshot_dict['DATA'] = DATA
                stats_snapshot_dict['LP_MODE'] = LP_MODE
                stats_snapshot_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
                stats_snapshot_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
                stats_snapshot_dict['run_idx'] = run_idx
                stats_snapshot_dict['snap_idx'] = snap_idx
                stats_list.append(stats_snapshot_dict)

                if run_idx == 0:
                    stats_header = ""
                    for key in list(stats_snapshot_dict.keys()):
                        stats_header += key + ","
                    if not os.path.isfile(stats_filename):
                        print(f"INFO: make a new stats file...")
                        with open(stats_filename, 'w') as writer:
                            writer.write(stats_header)
            else:
                print(f"DEBUG: File name: {iter_partial_filename}_src.npy")
                print(f"INFO: DATA: {DATA}, LP_MODE: {LP_MODE}: Run {run_idx}, Snapshot {snap_idx} does not exist!")
        
    stats_df = pd.read_csv(stats_filename)
    stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)])
    stats_df.to_csv(stats_filename, index=False)


def gen_stats_one_iteration(pred_dict, full_data):
    """
    generate statistics for one snapshot
    """
    src_list = pred_dict['sources']
    dst_list = pred_dict['destinations']
    tsp_list = pred_dict['timestamps']
    psc_list = pred_dict['pred_scores']
    lbl_list = pred_dict['labels']

    # regenerate positive data & negative data
    pos_mask = lbl_list == 1
    pos_data = Data(src_list[pos_mask], dst_list[pos_mask], tsp_list[pos_mask], [], lbl_list[pos_mask])  # idx_list[pos_mask], 
    neg_mask = lbl_list == 0 
    neg_data = Data(src_list[neg_mask], dst_list[neg_mask], tsp_list[neg_mask], [], lbl_list[neg_mask])  # idx_list[neg_mask], 

    all_edge_status = generate_snapshot_edge_status(pos_data, neg_data, full_data)
    hist_stat_list, use_in_eval_list = gen_meta_info_for_eval(src_list, dst_list, tsp_list, lbl_list, all_edge_status)

    res_df = pd.DataFrame(zip(src_list, dst_list, psc_list, lbl_list, hist_stat_list, use_in_eval_list),
                            columns=['source', 'destination', 'pred_score', 'label', 'hist', 'use_in_eval'])
    
    stats_dict = generate_MAR_stats(res_df)

    return stats_dict


def generate_MAR_stats(res_df, verbose=False):
    """
    generate MAR statistics given the results data frame
    """
    all_pos_e_rank = []
    distinct_src_list = np.unique(np.array(res_df['source']))
    for uniq_src in distinct_src_list:
        src_all_edges = res_df.loc[res_df['source'] == uniq_src]
        src_all_edges.sort_values(by="pred_score", ascending=False)
        pos_e_indices = np.array(list(np.where(src_all_edges["label"] == 1)[0]))
        src_pos_e_ranks = pos_e_indices + 1  # because rank should starts from 1, not 0
        all_pos_e_rank.append(src_pos_e_ranks)
    all_pos_e_rank = np.concatenate(all_pos_e_rank, axis=0)
    MAR = np.mean(all_pos_e_rank)
    if verbose:
        print("INFO: MAR: {}".format(MAR))

    stats_dict = {'MAR': MAR,
                    }
    return stats_dict


def gen_avg_perf(stats_df_filename, avg_stats_df_filename, DATA, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE):
    """
    generate average statistics across runs and snapshots
    """
    all_cols_w_values = ['MAR',
                         ]
    stats_df = pd.read_csv(stats_df_filename)
    setting_res = stats_df.loc[((stats_df['DATA'] == DATA) & (stats_df['LP_MODE'] == LP_MODE) & 
                                (stats_df['TR_NEG_SAMPLE'] == TR_NEG_SAMPLE) & (stats_df['TS_NEG_SAMPLE'] == TS_NEG_SAMPLE)),
                                all_cols_w_values]
    setting_avg_dict = dict(setting_res.mean())
    setting_avg_dict['DATA'] = DATA
    setting_avg_dict['LP_MODE'] = LP_MODE
    setting_avg_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
    setting_avg_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE

    if not os.path.isfile(avg_stats_df_filename):
        avg_stats_df = pd.DataFrame([setting_avg_dict])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_df_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame([setting_avg_dict])])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)


def main():
    """
    execution command:
        python interpret_MAR_EXH.py --seed 123 --n_runs 5 --lp_mode trans --data canVote --prefix tgn_attn
        
    """

    args, _ = get_args()
    DATA = args.data
    N_RUNS = args.n_runs
    LP_MODE = args.lp_mode  # 'trans' or 'induc'
    TR_NEG_SAMPLE = args.tr_neg_sample
    TS_NEG_SAMPLE = args.ts_neg_sample
    TR_RND_NE_RATIO = args.tr_rnd_ne_ratio
    TS_RND_NE_RATIO = args.ts_rnd_ne_ratio
    MODEL_NAME = args.prefix
    SEED = args.seed
    
    set_random_seed(SEED)

    NUM_SNAPSHOTS = {'canVote': 2,
                    'LegisEdgelist': 1,
                    'enron': 10,
                    'mooc': 10,
                    'reddit': 10,
                    'uci': 10,
                    'wikipedia': 10,
                    }

    partial_path = f"../EXH_dlp_stats/{MODEL_NAME}/"
    partial_filename = f"{MODEL_NAME}_{DATA}_TR_{TR_NEG_SAMPLE}_TS_{TS_NEG_SAMPLE}"

    stats_filename = f"{partial_path}/DLP_MAR_{MODEL_NAME}_{LP_MODE}.csv"
    stats_filename_avg = f"{partial_path}/DLP_MAR_{MODEL_NAME}_{LP_MODE}_avg.csv"

    print("="*150)
    print(f"INFO: METHOD: {MODEL_NAME}, DATA: {DATA}, N_RUNS: {N_RUNS}, LP_MODE: {LP_MODE}")
    print("="*150)

    # calculate MAR for the EXHaustive evaluation of dynamic link prediction task
    node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger=None, verbose=False)
    invoke_stats_generation(partial_path, partial_filename, stats_filename, full_data, MODEL_NAME, DATA, 
                            LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS)

    # generate average stats as well...
    gen_avg_perf(stats_filename, stats_filename_avg, DATA, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE)



if __name__ == '__main__':
    main()