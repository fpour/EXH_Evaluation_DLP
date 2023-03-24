"""
Post-process the link prediction statistics that were saved during the test phase

Date:
    - Jan. 04, 2023
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


num_entities_neg = 100

def get_args():
    parser = argparse.ArgumentParser('*** DLP Results Interpretation ***')
    # Related to process_stats
    parser.add_argument('--prefix', type=str, default='tgn_attn', choices=['tgn_attn', 'jodie_rnn', 'dyrep_rnn'], help='Model Prefix')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')
    parser.add_argument('--tr_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc'],
                        help='Strategy for the negative sampling at the training phase.')
    parser.add_argument('--ts_neg_sample', type=str, default='haphaz_rnd', choices=['rnd', 'haphaz_rnd', 'hist', 'induc', 'hitsK'],
                        help='Strategy for the negative edge sampling at the test phase.')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--avg_res', action='store_true', help='Compute and return the average performance results.')
    parser.add_argument('--eval_mode', type=str, default='STD', choices=['std', 'snapshot'], help='Evaluation mode.')
    parser.add_argument('--lp_mode', type=str, default='trans', choices=['trans', 'induc'],
                        help="Link prediction mode: transductive or inductive")
    parser.add_argument('--opt', type=str, default='gen',   # choices=['gen', 'avg', 'log', 'hits_at_k'], 
                        help='Generate new statistics or report average of the existing statistics.')
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

    
def gen_stats_one_iteration(pred_dict, EVAL_MODE, full_data):
    """
    generate statistics for one snapshot
    """
    src_list = pred_dict['sources']
    dst_list = pred_dict['destinations']
    tsp_list = pred_dict['timestamps']
    # idx_list = pred_dict['e_idxs']
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
    
    stats_dict = generate_results_stats(res_df, EVAL_MODE=EVAL_MODE)

    return stats_dict


def invoke_stats_generation(partial_path, partial_filename, stats_filename, full_data, DATA, 
                            EVAL_MODE, LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS):
    """
    Call one instance of the "gen_stats_one_iteration" method
    """
    stats_list = []
    for run_idx in range(N_RUNS):
        if EVAL_MODE == 'std':
            iter_partial_filename = f"{partial_path}/{partial_filename}_{run_idx}_{LP_MODE}"
            pred_dict = {'sources': np.load(f"{iter_partial_filename}_src.npy"),
                         'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                         'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                        #  'e_idxs': np.load(f"{iter_partial_filename}_e_idx.npy"),
                         'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                         'labels': np.load(f"{iter_partial_filename}_label.npy"),
                        }
            stats_snapshot_dict = gen_stats_one_iteration(pred_dict, EVAL_MODE, full_data)
            stats_snapshot_dict['DATA'] = DATA
            stats_snapshot_dict['EVAL_MODE'] = EVAL_MODE
            stats_snapshot_dict['LP_MODE'] = LP_MODE
            stats_snapshot_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
            stats_snapshot_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
            stats_list.append(stats_snapshot_dict)

            if run_idx == 0:
                stats_header = ""
                for key in list(stats_snapshot_dict.keys()):
                    stats_header += key + ","
                if not os.path.isfile(stats_filename):
                    with open(stats_filename, 'w') as writer:
                        writer.write(stats_header)

        elif EVAL_MODE == 'snapshot':
            for snap_idx in range(NUM_SNAPSHOTS[DATA]):
                iter_partial_filename = f"{partial_path}/snapshots/{partial_filename}_{run_idx}_{LP_MODE}_{snap_idx}"
                if os.path.isfile(f"{iter_partial_filename}_src.npy"):  # if the current snapshot info exists...
                    pred_dict = {'sources': np.load(f"{iter_partial_filename}_src.npy"),
                                'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                                'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                                # 'e_idxs': np.load(f"{iter_partial_filename}_e_idx.npy"),
                                'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                                'labels': np.load(f"{iter_partial_filename}_label.npy"),
                                }
                    stats_snapshot_dict = gen_stats_one_iteration(pred_dict, EVAL_MODE, full_data)
                    stats_snapshot_dict['DATA'] = DATA
                    stats_snapshot_dict['EVAL_MODE'] = EVAL_MODE
                    stats_snapshot_dict['LP_MODE'] = LP_MODE
                    stats_snapshot_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
                    stats_snapshot_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
                    stats_list.append(stats_snapshot_dict)

                    if run_idx == 0:
                        stats_header = ""
                        for key in list(stats_snapshot_dict.keys()):
                            stats_header += key + ","
                        if not os.path.isfile(stats_filename):
                            with open(stats_filename, 'w') as writer:
                                writer.write(stats_header)
                else:
                    print(f"INFO: DATA: {DATA}, EVAL_MODE: {EVAL_MODE}, LP_MODE: {LP_MODE}: Run {run_idx}, Snapshot {snap_idx} does not exist!")
        else:
            raise ValueError("INFO: Invalid evaluation mode!")
        
    stats_df = pd.read_csv(stats_filename)
    stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)])
    stats_df.to_csv(stats_filename, index=False)



def gen_res_df_for_computin_hits_at_k(pred_dict, full_data):
    """
    generate statistics for one snapshot
    """
    src_list = pred_dict['sources']
    dst_list = pred_dict['destinations']
    tsp_list = pred_dict['timestamps']
    # idx_list = pred_dict['e_idxs']
    psc_list = pred_dict['pred_scores']
    lbl_list = pred_dict['labels']

    # regenerate positive data & negative data
    pos_mask = lbl_list == 1
    pos_data = Data(src_list[pos_mask], dst_list[pos_mask], tsp_list[pos_mask], [], lbl_list[pos_mask])  # idx_list[pos_mask], 
    neg_mask = lbl_list == 0 
    neg_data = Data(src_list[neg_mask], dst_list[neg_mask], tsp_list[neg_mask], [], lbl_list[neg_mask])  # idx_list[neg_mask], 

    all_edge_status = generate_snapshot_edge_status(pos_data, neg_data, full_data)
    hist_stat_list, use_in_eval_list = gen_meta_info_for_eval(src_list, dst_list, tsp_list, lbl_list, all_edge_status)

    res_df = pd.DataFrame(zip(psc_list, lbl_list, hist_stat_list, use_in_eval_list),
                            columns=['pred_score', 'label', 'hist', 'use_in_eval'])

    return res_df


def invoke_hits_at_k_generation(partial_path, partial_filename, stats_filename, full_data, DATA, 
                                EVAL_MODE, LP_MODE, MODEL_NAME, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS, k_list):
    """
    Call one instance of the "gen_stats_one_iteration" method
    """
    stats_list = []
    for run_idx in range(N_RUNS):
        if EVAL_MODE == 'std':
            iter_partial_filename = f"{partial_path}/{partial_filename}_{run_idx}_{LP_MODE}"
            pred_dict = {'sources': np.load(f"{iter_partial_filename}_src.npy"),
                         'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                         'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                        #  'e_idxs': np.load(f"{iter_partial_filename}_e_idx.npy"),
                         'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                         'labels': np.load(f"{iter_partial_filename}_label.npy"),
                        }
            res_df = gen_res_df_for_computin_hits_at_k(pred_dict, full_data)

        elif EVAL_MODE == 'snapshot':
            res_df_list = []
            for snap_idx in range(NUM_SNAPSHOTS[DATA]):
                iter_partial_filename = f"{partial_path}/snapshots/{partial_filename}_{run_idx}_{LP_MODE}_{snap_idx}"
                if os.path.isfile(f"{iter_partial_filename}_src.npy"):  # if the current snapshot info exists...
                    pred_dict = {'sources': np.load(f"{iter_partial_filename}_src.npy"),
                                'destinations': np.load(f"{iter_partial_filename}_dst.npy"),
                                'timestamps': np.load(f"{iter_partial_filename}_ts.npy"),
                                # 'e_idxs': np.load(f"{iter_partial_filename}_e_idx.npy"),
                                'pred_scores': np.load(f"{iter_partial_filename}_pred_score.npy"),
                                'labels': np.load(f"{iter_partial_filename}_label.npy"),
                                }
                    res_df = gen_res_df_for_computin_hits_at_k(pred_dict, full_data)
                    res_df_list.append(res_df)
                else:
                    print(f"INFO: DATA: {DATA}, EVAL_MODE: {EVAL_MODE}, LP_MODE: {LP_MODE}: Run {run_idx}, Snapshot {snap_idx} does not exist!")
            # append all snapshots res_df 
            res_df = pd.concat(res_df_list, ignore_index=True)
        else:
            raise ValueError("INFO: Invalid evaluation mode!")
        
        for k in k_list:
            stats_dict = hits_at_k(res_df, k=k, mode=EVAL_MODE)
            stats_dict['DATA'] = DATA
            stats_dict['MODEL'] = MODEL_NAME
            stats_dict['EVAL_MODE'] = EVAL_MODE
            stats_dict['LP_MODE'] = LP_MODE
            stats_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
            stats_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
            stats_dict['k'] = k
            stats_list.append(stats_dict)

        if run_idx == 0:
            stats_header = ""
            for key in list(stats_dict.keys()):
                stats_header += key + ","
            if not os.path.isfile(stats_filename):
                with open(stats_filename, 'w') as writer:
                    writer.write(stats_header)

    stats_df = pd.read_csv(stats_filename)
    stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)])
    stats_df.to_csv(stats_filename, index=False)


def get_hits_at_k_avg(stats_filename, avg_stats_filename, model, data, eval_mode, lp_mode, tr_neg_sample, ts_neg_sample, k_list):
    metrics = ['all_hits_at_k', 'hist_hits_at_k', 'new_hits_at_k']
    stats_df = pd.read_csv(stats_filename)
    avg_res_list = []
    for k in k_list:
        selected_rows = stats_df.loc[(stats_df['MODEL'] == model) & (stats_df['DATA'] == data) & (stats_df['k'] == k)
                                    & (stats_df['EVAL_MODE'] == eval_mode) & (stats_df['LP_MODE'] == lp_mode)
                                    & (stats_df['TR_NEG_SAMPLE'] == tr_neg_sample) & (stats_df['TS_NEG_SAMPLE'] == ts_neg_sample), metrics]
        avg_res = dict(selected_rows.mean())
        avg_res['MODEL'] = model
        avg_res['DATA'] = data
        avg_res['EVAL_MODE'] = eval_mode
        avg_res['LP_MODE'] = lp_mode
        avg_res['TR_NEG_SAMPLE'] = tr_neg_sample
        avg_res['TS_NEG_SAMPLE'] = ts_neg_sample
        avg_res['k'] = k
        avg_res_list.append(avg_res)

    if not os.path.isfile(avg_stats_filename):
        avg_stats_df = pd.DataFrame(avg_res_list)
        avg_stats_df.to_csv(avg_stats_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame(avg_res_list)])
        avg_stats_df.to_csv(avg_stats_filename, index=False)


def get_avg_based_on_log(stats_filename, avg_stats_filename, model, data, tr_neg_sample, ts_neg_sample):
    """
    get the average of the results generated besides the log files
    """
    metrics = ['ap', 'auc', 'opt_thr_auc', 'aupr', 'opt_thr_aupr', 'acc', 'prec', 'rec', 'f1']
    stats_df = pd.read_csv(stats_filename)
    selected_rows = stats_df.loc[(stats_df['model'] == model) & (stats_df['data'] == data) 
                                 & (stats_df['tr_neg_sample'] == tr_neg_sample) & (stats_df['ts_neg_sample'] == ts_neg_sample), metrics]
    avg_res = dict(selected_rows.mean())
    avg_res['MODEL'] = model
    avg_res['DATA'] = data
    avg_res['TR_NEG_SAMPLE'] = tr_neg_sample
    avg_res['TS_NEG_SAMPLE'] = ts_neg_sample

    if not os.path.isfile(avg_stats_filename):
        avg_stats_df = pd.DataFrame([avg_res])
        avg_stats_df.to_csv(avg_stats_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame([avg_res])])
        avg_stats_df.to_csv(avg_stats_filename, index=False)


def invoke_eval_histK_OGB(partial_path, partial_name, stats_filename, DATA, 
                             EVAL_MODE, LP_MODE, MODEL_NAME, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE):
    """
    Call one instance of the "gen_stats_one_iteration" method
    """
    stats_list = []
    for run_idx in range(N_RUNS):
        iter_partial_filename = f"{partial_path}/{TS_NEG_SAMPLE}/{partial_name}_{run_idx}_{LP_MODE}"
        # print("DEBUG: iter_partial_filename:", iter_partial_filename)
        labels = np.load(f"{iter_partial_filename}_label.npy")
        pred_scores = np.load(f"{iter_partial_filename}_pred_score.npy")
        
        stats_dict = eval_mrr_OGB(labels, pred_scores, num_entities_neg)
        # for k in k_list:
        #     hitsK = eval_hits_OGB(labels, pred_scores, k)
        #     stats_dict[f'Hits@{k}'] = hitsK
        stats_dict['DATA'] = DATA
        stats_dict['MODEL'] = MODEL_NAME
        stats_dict['EVAL_MODE'] = EVAL_MODE
        stats_dict['LP_MODE'] = LP_MODE
        stats_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
        stats_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE
        stats_list.append(stats_dict)

        if run_idx == 0:
            stats_header = ""
            for key in list(stats_dict.keys()):
                stats_header += key + ","
            if not os.path.isfile(stats_filename):
                with open(stats_filename, 'w') as writer:
                    writer.write(stats_header)

    stats_df = pd.read_csv(stats_filename)
    stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)])
    stats_df.to_csv(stats_filename, index=False)


def get_HitsK_avg_OGB(stats_filename, avg_stats_filename, model, data, eval_mode, lp_mode, tr_neg_sample, ts_neg_sample, k_list):
    metrics = ['hits@1', 'hits@3', 'hits@10', 'hits@50', 'hits@100', 'mrr']   # + [f'Hits@{k}' for k in k_list]
    stats_df = pd.read_csv(stats_filename)
    avg_res_list = []
    selected_rows = stats_df.loc[(stats_df['MODEL'] == model) & (stats_df['DATA'] == data)
                                & (stats_df['EVAL_MODE'] == eval_mode) & (stats_df['LP_MODE'] == lp_mode)
                                & (stats_df['TR_NEG_SAMPLE'] == tr_neg_sample) & (stats_df['TS_NEG_SAMPLE'] == ts_neg_sample), metrics]
    avg_res = dict(selected_rows.mean())
    avg_res['MODEL'] = model
    avg_res['DATA'] = data
    avg_res['EVAL_MODE'] = eval_mode
    avg_res['LP_MODE'] = lp_mode
    avg_res['TR_NEG_SAMPLE'] = tr_neg_sample
    avg_res['TS_NEG_SAMPLE'] = ts_neg_sample
    avg_res_list.append(avg_res)

    if not os.path.isfile(avg_stats_filename):
        avg_stats_df = pd.DataFrame(avg_res_list)
        avg_stats_df.to_csv(avg_stats_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame(avg_res_list)])
        avg_stats_df.to_csv(avg_stats_filename, index=False)

def main():
    """
    execution command:
        python interpret_scores_LP.py --seed 123 --n_runs 5 --eval_mode std --lp_mode trans --data canVote --prefix tgn_attn --opt gen
        python interpret_scores_LP.py --seed 123 --n_runs 1 --eval_mode std --lp_mode trans --data canVote --prefix tgn_attn --opt OGB --ts_neg_sample hitsK
    """

    args, _ = get_args()
    DATA = args.data
    N_RUNS = args.n_runs
    OPT = args.opt
    EVAL_MODE = args.eval_mode  # 'STD' or 'SNAPSHOT'
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

    partial_path = f"LP_stats/{DATA}"
    # partial_name = f"{MODEL_NAME}_{DATA}_{NEG_SAMPLE}"
    partial_name = f"{MODEL_NAME}_{DATA}_TR_{TR_NEG_SAMPLE}_TS_{TS_NEG_SAMPLE}"

    stats_filename = f"LP_stats/Interpreted_Stats/LP_pred_scores_{MODEL_NAME}_{EVAL_MODE}.csv"
    avg_stats_df_filename = f"LP_stats/Interpreted_Stats/LP_pred_scores_{MODEL_NAME}_{EVAL_MODE}_avg.csv"

    print("="*150)
    print(f"INFO: METHOD: {MODEL_NAME}, DATA: {DATA}, OPT: {OPT}, TR_NEG_SAMPLE: {TR_NEG_SAMPLE}, TS_NEG_SAMPLE: {TS_NEG_SAMPLE}, N_RUNS: {N_RUNS}, EVAL_MODE: {EVAL_MODE}, LP_MODE: {LP_MODE}")
    print("="*150)

    if OPT == 'avg':
        print("INFO: *** Reporting the average statistics ***")
        gen_avg_perf(stats_filename, avg_stats_df_filename, DATA, EVAL_MODE, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE)

    elif OPT == 'gen':
        print("INFO: *** Generating statistics ***")
        node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger=None, verbose=False)
        invoke_stats_generation(partial_path, partial_name, stats_filename, full_data, DATA, 
                                EVAL_MODE, LP_MODE, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS)

    elif OPT == 'log':
        if LP_MODE == 'induc':
            keyword = 'INDUC'
        elif LP_MODE == 'trans':
            keyword = 'TRANS'
        stats_filename = f'LP_stats/STD_pred_{keyword}.csv'
        avg_stats_filename = f'LP_stats/Interpreted_Stats/STD_pred_{keyword}_avg.csv'
        get_avg_based_on_log(stats_filename, avg_stats_filename, MODEL_NAME, DATA, TR_NEG_SAMPLE, TS_NEG_SAMPLE)

    elif OPT == 'hits_at_k':
        k_list = [10, 50, 100, 500, 1000, 10000, 50000, 100000, 1000000]
        print(f"INFO: *** Computing Hits@k ***")
        stats_filename = f'LP_stats/Interpreted_Stats/Hits_at_k_{EVAL_MODE}.csv'
        avg_stats_filename = f'LP_stats/Interpreted_Stats/Hits_at_k_{EVAL_MODE}_avg.csv'
        node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data = get_data(DATA, args, logger=None, verbose=False)
        invoke_hits_at_k_generation(partial_path, partial_name, stats_filename, full_data, DATA, 
                                EVAL_MODE, LP_MODE, MODEL_NAME, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE, NUM_SNAPSHOTS, k_list)
        get_hits_at_k_avg(stats_filename, avg_stats_filename, MODEL_NAME, DATA, EVAL_MODE, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE, k_list)

    elif OPT == 'OGB':
        stats_filename = f'LP_stats/Interpreted_Stats/Hits_at_k_{EVAL_MODE}_OGB.csv'
        avg_stats_filename = f'LP_stats/Interpreted_Stats/Hits_at_k_{EVAL_MODE}_OGB_avg.csv'
        invoke_eval_histK_OGB(partial_path, partial_name, stats_filename, DATA, 
                             EVAL_MODE, LP_MODE, MODEL_NAME, N_RUNS, TR_NEG_SAMPLE, TS_NEG_SAMPLE)
        get_HitsK_avg_OGB(stats_filename, avg_stats_filename, MODEL_NAME, DATA, EVAL_MODE, LP_MODE, 
                          TR_NEG_SAMPLE, TS_NEG_SAMPLE, k_list)

    else:
        raise ValueError("INFO: Invalid option!!!")



if __name__ == '__main__':
    main()
    


