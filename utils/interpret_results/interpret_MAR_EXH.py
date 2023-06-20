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


num_entities_neg = 100

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
            iter_partial_filename = f"{partial_path}/{DATA}_snapshots/{MODEL_NAME}_{partial_filename}_{run_idx}_{LP_MODE}_{snap_idx}"
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
    distinct_src_list = np.unique(np.array(res_df['source']))
    for uniq_src in distinct_src_list:
        src_all_edges = res_df.loc[res_df['source'] == uniq_src]
        src_all_edges.sort_values(by="pred_score", ascending=False)

    
    !!! I was here!

    stats_dic = {}
    # some statistics about the test set edge list
    # count number of repeated edges in each categories
    pos_hist_e = res_df.loc[(res_df['hist'] == 1) & (res_df['label'] == 1)]
    pos_new_e = res_df.loc[(res_df['hist'] == 0) & (res_df['label'] == 1)]
    neg_hist_e = res_df.loc[(res_df['hist'] == 1) & (res_df['label'] == 0)]
    neg_new_e = res_df.loc[(res_df['hist'] == 0) & (res_df['label'] == 0)]

    stats_dic['n_pos_hist'] = pos_hist_e.shape[0]
    stats_dic['n_pos_new'] = pos_new_e.shape[0]
    stats_dic['n_neg_hist'] = neg_hist_e.shape[0]
    stats_dic['n_neg_new'] = neg_new_e.shape[0]

    # compute the number of repetitions of the same edges through test phase
    stats_dic['n_repeated_pos_hist'] = count_repetition(pos_hist_e)
    stats_dic['n_repeated_pos_new'] = count_repetition(pos_new_e)
    stats_dic['n_repeated_neg_hist'] = count_repetition(neg_hist_e)
    stats_dic['n_repeated_neg_new'] = count_repetition(neg_new_e)

    if EVAL_MODE == 'snapshot':
        print("INFO: *** SNAPSHOT-based ***")
        print("INFO: NOTICE: Only Distinct edges are used for performance evaluation!")
        res_df = res_df.loc[res_df['use_in_eval'] == 1]  # only the one that we want to use for evaluations

        stats_dic['n_pos_hist_used_in_eval'] = res_df.loc[(res_df['hist'] == 1) & (res_df['label'] == 1)].shape[0]
        stats_dic['n_pos_new_used_in_eval'] = res_df.loc[(res_df['hist'] == 0) & (res_df['label'] == 1)].shape[0]
        stats_dic['n_neg_hist_used_in_eval'] = res_df.loc[(res_df['hist'] == 1) & (res_df['label'] == 0)].shape[0]
        stats_dic['n_neg_new_used_in_eval'] = res_df.loc[(res_df['hist'] == 0) & (res_df['label'] == 0)].shape[0]


        print("INFO: Some statistics about the snapshot:")
        print(f"\tINFO: POS_HIST: Total: {stats_dic['n_pos_hist']}\tRepeated: {stats_dic['n_repeated_pos_hist']}\tUsed in eval.: {stats_dic['n_pos_hist_used_in_eval']}")
        print(
            f"\tINFO: POS_LNEW: Total: {stats_dic['n_pos_new']}\tRepeated: {stats_dic['n_repeated_pos_new']}\tUsed in eval.: {stats_dic['n_pos_new_used_in_eval']}")
        print(
            f"\tINFO: NEG_HIST: Total: {stats_dic['n_neg_hist']}\tRepeated: {stats_dic['n_repeated_neg_hist']}\tUsed in eval.: {stats_dic['n_neg_hist_used_in_eval']}")
        print(
            f"\tINFO: NEG_LNEW: Total: {stats_dic['n_neg_new']}\tRepeated: {stats_dic['n_repeated_neg_new']}\tUsed in eval.: {stats_dic['n_neg_new_used_in_eval']}")

    # divide into historical and new edges
    # historical edges
    hist_e_pred = np.array(res_df.loc[res_df['hist'] == 1, 'pred_score'].tolist())
    hist_e_label = np.array(res_df.loc[res_df['hist'] == 1, 'label'].tolist())
    stats_dic['hist_e_PRAUC'], stats_dic['hist_e_AUC'], stats_dic['hist_e_AP'] = compute_metrics(hist_e_label, hist_e_pred)

    # new edges
    new_e_pred = np.array(res_df.loc[res_df['hist'] == 0, 'pred_score'].tolist())
    new_e_label = np.array(res_df.loc[res_df['hist'] == 0, 'label'].tolist())
    stats_dic['new_e_PRAUC'], stats_dic['new_e_AUC'], stats_dic['new_e_AP'] = compute_metrics(new_e_label, new_e_pred)

    # calculate GMAUC
    try: 
        stats_dic['GMAUC'] = np.sqrt(((stats_dic['new_e_PRAUC'] - (float(stats_dic['n_pos_new']) /
                                                               float((stats_dic['n_pos_new'] + stats_dic['n_neg_new'])))) /
                                  (1 - (float(stats_dic['n_pos_new']) / (float(stats_dic['n_pos_new']
                                                                                 + stats_dic['n_neg_new']))))) * 2 * (
                                         stats_dic['hist_e_AUC'] - 0.5))
    except ZeroDivisionError:
        stats_dic['GMAUC'] = 'NA'

    # all categories together
    all_e_label = np.array(res_df['label'].tolist())
    all_e_pred = np.array(res_df['pred_score'].tolist())
    stats_dic['all_e_PRAUC'], stats_dic['all_e_AUC'], stats_dic['all_e_AP'] = compute_metrics(all_e_label, all_e_pred)
    # stats_dic['all_e_AP'] = average_precision_score(all_e_label, all_e_pred)

    if verbose:
        # print out the results
        print("--------------------------------------------------------------------")
        print(f"NUM_POS_HIST: {stats_dic['n_pos_hist']}; \tNUM_REPETITION: {stats_dic['n_repeated_pos_hist']}")
        print(f"NUM_NEG_HIST: {stats_dic['n_neg_hist']}; \tNUM_REPETITION: {stats_dic['n_repeated_neg_hist']}")
        print(f"NUM_POS_NEW: {stats_dic['n_pos_new']}; \tNUM_REPETITION: {stats_dic['n_repeated_pos_new']}")
        print(f"NUM_NEG_NEW: {stats_dic['n_neg_new']}; \tNUM_REPETITION: {stats_dic['n_repeated_neg_new']}")
        print("*** HISTORICAL EDGES:")
        print(f"\tHIST_AUC: {stats_dic['hist_e_AUC']}, HIST_PRAUC: {stats_dic['hist_e_PRAUC']}")
        print("*** NEW EDGES:")
        print(f"\tNEW_AUC: {stats_dic['new_e_AUC']}, NEW_PRAUC: {stats_dic['new_e_PRAUC']}")
        print("*** Combined Metric (NEW_PRAUC & HIST_AUC):")
        print(f"\tGMAUC: {stats_dic['GMAUC']}")
        print("*** ALL TEST EDGES TOGETHER:")
        print(f"\tALL_TEST_AUC: {stats_dic['all_e_AUC']}, ALL_TEST_PRAUC: {stats_dic['all_e_PRAUC']}, ALL_TEST_AP: {stats_dic['all_e_AP']}")
        print("--------------------------------------------------------------------")

    return stats_dic



def main():
    """
    execution command:
        python interpret_scores_LP.py --seed 123 --n_runs 5 --eval_mode std --lp_mode trans --data canVote --prefix tgn_attn --opt gen
        python interpret_scores_LP.py --seed 123 --n_runs 1 --eval_mode std --lp_mode trans --data canVote --prefix tgn_attn --opt OGB --ts_neg_sample hitsK
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

    partial_path = f"./EXH_dlp_stats/{MODEL_NAME}/"
    partial_name = f"{MODEL_NAME}_{DATA}_TR_{TR_NEG_SAMPLE}_TS_{TS_NEG_SAMPLE}"

    stats_filename = f"LP_stats/Interpreted_Stats/LP_pred_scores_{MODEL_NAME}_{EVAL_MODE}.csv"
    stats_filename = f"{partial_path}/{MODEL_NAME}/DLP_MAR_{MODEL_NAME}.csv"

    print("="*150)
    print(f"INFO: METHOD: {MODEL_NAME}, DATA: {DATA}, N_RUNS: {N_RUNS}, LP_MODE: {LP_MODE}")
    print("="*150)

    # calculate MAR for the EXHaustive evaluation of dynamic link prediction task




if __name__ == '__main__':
    main()