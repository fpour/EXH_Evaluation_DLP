"""
Process the results

Date:
    - Jan. 05

"""

import numpy as np
import pandas as pd
from utils.test_data_generate import get_unique_edges
from sklearn.metrics import *
import os.path

# ======================================================================================================
# ======================================= Edge Status Processing =======================================
# ======================================================================================================

def get_last_positive_edge(pos_e_data):
    """
    return the last latest positive edge that has been observed if there are repetition of the same positive edge
    """
    sources = pos_e_data.sources
    destinations = pos_e_data.destinations
    timestamps = pos_e_data.timestamps
    distinct_pos_edges = {}
    # removing redundant edges
    for idx, (src, dst) in enumerate(zip(sources, destinations)):
        if (src, dst) not in distinct_pos_edges:
            distinct_pos_edges[(src, dst)] = {'timestamp': timestamps[idx],
                                              }
        else:
            if distinct_pos_edges[(src, dst)]['timestamp'] < timestamps[idx]:
                distinct_pos_edges[(src, dst)]['timestamp'] = timestamps[idx]
    return distinct_pos_edges


def generate_snapshot_edge_status(pos_data, neg_data, full_data):
    """
    generate teh edge status for all the edges in one snapshot
    """
    # get the historical edges: seen before THIS current snapshot
    hist_masks = full_data.timestamps < min(pos_data.timestamps)  # NOTE: memory of the historical edges is updating during the test phase!!!
    hist_srcs = full_data.sources[hist_masks]
    hist_dst = full_data.destinations[hist_masks]
    hist_e = get_unique_edges(hist_srcs, hist_dst)

    distinct_pos_edges = get_last_positive_edge(pos_data)

    all_edge_status = {}
    pos_last_ts_seen = {}

    # positive edges
    for pos_idx, (src, dst, ts) in enumerate(zip(pos_data.sources, pos_data.destinations, pos_data.timestamps)):
        last_ts = distinct_pos_edges[(src, dst)]['timestamp']
        if ((src, dst) in hist_e) and (ts == last_ts):
            all_edge_status[(src, dst, ts)] = {'hist': 1,
                                               'use_in_eval': 1,
                                               'last_ts': last_ts,
                                               }
        elif ((src, dst) in hist_e) and (ts < last_ts):  # a repeated historical positive edge
            all_edge_status[(src, dst, ts)] = {'hist': 1,
                                               'use_in_eval': 0,
                                               'last_ts': last_ts,
                                               }
        elif ((src, dst) not in hist_e) and (ts == last_ts):
            all_edge_status[(src, dst, ts)] = {'hist': 0,
                                               'use_in_eval': 1,
                                               'last_ts': last_ts,
                                               }
        elif ((src, dst) not in hist_e) and (ts < last_ts):  # a repeated new positive edge
            all_edge_status[(src, dst, ts)] = {'hist': 0,
                                               'use_in_eval': 0,
                                               'last_ts': last_ts,
                                               }
        else:
            raise ValueError("ERROR: This case is impossible; there is a major problem with the logic!")

    # negative edges
    for neg_idx, (src, dst, ts) in enumerate(zip(neg_data.sources, neg_data.destinations, neg_data.timestamps)):
        if (src, dst) in hist_e:
            all_edge_status[(src, dst, ts)] = {'hist': 1,
                                               'use_in_eval': 1,
                                               'last_ts': ts,
                                               }
        else:
            all_edge_status[(src, dst, ts)] = {'hist': 0,
                                               'use_in_eval': 1,
                                               'last_ts': ts,
                                               }
    
    return all_edge_status


def gen_meta_info_for_eval(src_list, dst_list, ts_list, lbl_list, all_edge_status):
    """
    generate the meta information required for prediction processing
    """
    e_hist_stat_list, e_use_in_eval_list = [], []
    # positive edges
    pos_dist_at_last_ts = {}
    for src, dst, ts, lbl in zip(src_list, dst_list, ts_list, lbl_list):
        e_hist_stat_list.append(all_edge_status[(src, dst, ts)]['hist'])
        if lbl == 1:  # a positive edge
            # fill the 'use_in_eval' attribute
            if ts == all_edge_status[(src, dst, ts)]['last_ts']:
                if (src, dst) not in pos_dist_at_last_ts:
                    pos_dist_at_last_ts[(src, dst)] = 1
                    e_use_in_eval_list.append(all_edge_status[(src, dst, ts)]['use_in_eval'])  # should append '1'
                else:
                    e_use_in_eval_list.append(0)
            else:
                e_use_in_eval_list.append(all_edge_status[(src, dst, ts)]['use_in_eval'])  # should append '0'
        else:
            e_use_in_eval_list.append(all_edge_status[(src, dst, ts)]['use_in_eval'])

    return e_hist_stat_list, e_use_in_eval_list



# ======================================================================================================
# ======================================= Results interpretation =======================================
# ======================================================================================================

def count_repetition(res_df_subset):
    """
    count whether there is any repetition of the same edge through time
    """
    src_list, dst_list = res_df_subset['source'].tolist(), res_df_subset['destination'].tolist()
    unique_e_dict = {}
    num_repeated_e = 0
    for src, dst in zip(src_list, dst_list):
        if (src, dst) not in unique_e_dict:
            unique_e_dict[(src, dst)] = 1
        else:
            num_repeated_e += 1
    return num_repeated_e


def compute_metrics(labels, pred_scores):
    num_unique_lbls = len(np.unique(np.array(labels)))
    if num_unique_lbls == 2:
        prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(labels, pred_scores)
        PRAUC = auc(rec_pr_curve, prec_pr_curve)
        AUC = roc_auc_score(labels, pred_scores)
        AP = average_precision_score(labels, pred_scores)
    else:
        AUC = 0
        PRAUC = 0
        AP = 0
        print("ATTENTION: Only one class present in y_true >>> Returns 0!")
    return PRAUC, AUC, AP



def generate_results_stats(res_df, EVAL_MODE, verbose=False):
    """
    generate statistics given the results data frame
    """
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

def gen_avg_perf(all_stats_df_filename, avg_stats_df_filename, DATA, EVAL_MODE, LP_MODE, TR_NEG_SAMPLE, TS_NEG_SAMPLE):


    all_cols_w_values = ['hist_e_PRAUC', 'hist_e_AUC', 'hist_e_AP', 'new_e_PRAUC', 'new_e_AUC', 'new_e_AP', 'GMAUC', 
                         'all_e_PRAUC', 'all_e_AUC', 'all_e_AP', 
                         'n_pos_hist', 'n_pos_new',
                         'n_neg_hist', 'n_neg_new',
                         'n_repeated_pos_hist', 'n_repeated_pos_new',
                         'n_repeated_neg_hist', 'n_repeated_neg_new',
                         ]
    if EVAL_MODE == 'snapshot':
        all_cols_w_values = all_cols_w_values + ['n_pos_hist_used_in_eval', 'n_pos_new_used_in_eval', 'n_neg_hist_used_in_eval', 'n_neg_new_used_in_eval']

    stats_df = pd.read_csv(all_stats_df_filename)
    setting_res = stats_df.loc[((stats_df['DATA'] == DATA) & (stats_df['LP_MODE'] == LP_MODE) & (stats_df['EVAL_MODE'] == EVAL_MODE) &
                                (stats_df['TR_NEG_SAMPLE'] == TR_NEG_SAMPLE) & (stats_df['TS_NEG_SAMPLE'] == TS_NEG_SAMPLE)),
                                all_cols_w_values]
    setting_avg_dict = dict(setting_res.mean())
    setting_avg_dict['DATA'] = DATA
    setting_avg_dict['LP_MODE'] = LP_MODE
    setting_avg_dict['EVAL_MODE'] = EVAL_MODE
    setting_avg_dict['TR_NEG_SAMPLE'] = TR_NEG_SAMPLE
    setting_avg_dict['TS_NEG_SAMPLE'] = TS_NEG_SAMPLE

    if not os.path.isfile(avg_stats_df_filename):
        avg_stats_df = pd.DataFrame([setting_avg_dict])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)
    else:
        avg_stats_df = pd.read_csv(avg_stats_df_filename)
        avg_stats_df = pd.concat([avg_stats_df, pd.DataFrame([setting_avg_dict])])
        avg_stats_df.to_csv(avg_stats_df_filename, index=False)



def hits_at_k(res_df, k=100, mode='STD'):
    """
    generate Hits@K for standard dynamic link evaluation
    """
    if mode == 'snapshot':
        res_df = res_df.loc[res_df['use_in_eval'] == 1]  # only the one that we want to use for evaluations

    sorted_res_df = res_df.sort_values(by=['pred_score'], ascending=False)

    # all edges
    all_hits_at_k = len([lbl for lbl in sorted_res_df['label'].tolist()[0: k] if lbl == 1])*1.0/k


    # historical edges
    hist_res_df = res_df.loc[res_df['hist'] == 1]
    sorted_hist_res_df = hist_res_df.sort_values(by=['pred_score'], ascending=False)
    hist_hits_at_k = len([lbl for lbl in sorted_hist_res_df['label'].tolist()[0: k] if lbl == 1])*1.0/k

    # new edges
    new_res_df =res_df.loc[res_df['hist'] == 0]
    sorted_new_res_df = new_res_df.sort_values(by=['pred_score'], ascending=False)
    new_hits_at_k = len([lbl for lbl in sorted_new_res_df['label'].tolist()[0: k] if lbl == 1])*1.0/k

    stats_dict = {'all_hits_at_k': all_hits_at_k,
                  'hist_hits_at_k': hist_hits_at_k,
                  'new_hits_at_k': new_hits_at_k
                  }

    return stats_dict


# def eval_hits_OGB(labels, pred_scores, K):
#     """
#     Evaluate Hits@K according to the OGB: 
#     https://github.com/snap-stanford/ogb/blob/d37cffa2e2cde531ca7b7e75800d331ed1e738a6/ogb/linkproppred/evaluate.py#L214
#     @TODO: should be revised, does not return correct values
#     """
#     # separate positive and negative predictions
#     neg_lbl_idx = labels == 0
#     y_pred_neg = pred_scores[neg_lbl_idx]

#     pos_lbl_idx = labels == 1
#     y_pred_pos = pred_scores[pos_lbl_idx]

#     kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
#     hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

#     return hitsK


def eval_mrr_OGB(labels, pred_scores, num_entities_neg):
    """
    Evaluate MRR according to the OGB: 
    """
    # separate positive and negative predictions
    assert len(labels) == len(pred_scores)

    neg_lbl_idx = labels == 0
    y_pred_neg_list = pred_scores[neg_lbl_idx]

    pos_lbl_idx = labels == 1
    y_pred_pos_list = pred_scores[pos_lbl_idx]

    hits1_list, hits3_list, hits10_list, hits50_list, hits100_list, mrr_list = [], [], [], [], [], []
    for pos_idx, y_pred_pos in enumerate(y_pred_pos_list):
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        n_s_idx = pos_idx * num_entities_neg
        n_e_idx = min(n_s_idx + num_entities_neg, len(y_pred_neg_list))
        y_pred_neg = y_pred_neg_list[n_s_idx: n_e_idx]

        optimistic_rank = (y_pred_neg >= y_pred_pos).sum()  # dim=1
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum()  # dim=1
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1


        hits1_list.append((ranking_list <= 1).astype(np.float32))
        hits3_list.append((ranking_list <= 3).astype(np.float32))
        hits10_list.append((ranking_list <= 10).astype(np.float32))
        hits50_list.append((ranking_list <= 50).astype(np.float32))
        hits100_list.append((ranking_list <= 100).astype(np.float32))
        mrr_list.append(1./ranking_list.astype(np.float32))


    return {'hits@1': np.mean(hits1_list) if len(hits1_list) >= 1 else 0,
            'hits@3': np.mean(hits3_list) if len(hits3_list) >= 1 else 0,
            'hits@10': np.mean(hits10_list) if len(hits10_list) >= 1 else 0,
            'hits@50': np.mean(hits50_list) if len(hits50_list) >= 1 else 0,
            'hits@100': np.mean(hits100_list) if len(hits100_list) >= 1 else 0,
            'mrr': np.mean(mrr_list) if len(mrr_list) >= 1 else 0,
            }


    