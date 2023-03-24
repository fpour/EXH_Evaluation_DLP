"""
Functions required for evaluation of the link prediction task
"""

import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import *



def get_metric_for_threshold(y_true, y_pred_score, threshold):
    """
    compute measures for a specific threshold
    """
    perf_measures = {}
    y_pred_label = y_pred_score > threshold
    perf_measures['acc'] = accuracy_score(y_true, y_pred_label)
    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=1)
    perf_measures['prec'] = prec
    perf_measures['rec'] = rec
    perf_measures['f1'] = f1
    return perf_measures


def compute_perf_metrics(y_true, y_pred_score):
    """
    compute extra performance measures
    """
    perf_dict = {}
    # find optimal threshold of au-roc
    perf_dict['ap'] = average_precision_score(y_true, y_pred_score)

    perf_dict['auc'] = roc_auc_score(y_true, y_pred_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr_auroc = roc_thresholds[opt_idx]
    perf_dict['opt_thr_auc'] = opt_thr_auroc

    prec_pr_curve, rec_pr_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_score)
    perf_dict['aupr'] = auc(rec_pr_curve, prec_pr_curve)
    fscore = (2 * prec_pr_curve * rec_pr_curve) / (prec_pr_curve + rec_pr_curve)
    opt_idx = np.argmax(fscore)
    opt_thr_aupr = pr_thresholds[opt_idx]
    perf_dict['opt_thr_aupr'] = opt_thr_aupr

    # threshold = 0.5: it is assumed that the threshold should be set before the test phase
    perf_half_dict = get_metric_for_threshold(y_true, y_pred_score, 0.5)
    perf_dict['acc'] = perf_half_dict['acc']
    perf_dict['prec'] = perf_half_dict['prec']
    perf_dict['rec'] = perf_half_dict['rec']
    perf_dict['f1'] = perf_half_dict['f1']

    return perf_dict


def eval_one_epoch_original(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)


def eval_link_pred(model, sampler, data, stats_filename=None, batch_size=32):
    """
    Evaluate the link prediction task
    """
    if stats_filename is not None:
        print("INFO: Test edge evaluation statistics are saved at {}".format(stats_filename))
    pred_score_list, true_label_list = [], []
    src_agg, dst_agg, ts_agg, e_idx_agg = [], [], [], []
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            # positive edges
            src_l_cut = data.sources[s_idx:e_idx]
            dst_l_cut = data.destinations[s_idx:e_idx]
            ts_l_cut = data.timestamps[s_idx:e_idx]
            e_l_cut = data.edge_idxs[s_idx:e_idx]

            # negative edges
            size = len(src_l_cut)
            neg_hist_ne_source, neg_hist_ne_dest, neg_rnd_source, neg_rnd_dest = sampler.sample(size, ts_l_cut[0], ts_l_cut[-1])

            src_l_fake = np.concatenate([neg_hist_ne_source, neg_rnd_source], axis=0)
            dst_l_fake = np.concatenate([neg_hist_ne_dest, neg_rnd_dest], axis=0)                

            # edge prediction
            pos_prob = model.contrast_modified(src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, pos_edge=True, test=True)
            if sampler.neg_sample == 'haphaz_rnd':
                src_l_fake = src_l_cut
                neg_prob = model.contrast_modified(src_l_fake, dst_l_fake, ts_l_cut, e_l_cut, pos_edge=False, test=True)
            else:
                neg_prob = model.contrast_modified(src_l_fake, dst_l_fake, ts_l_cut, e_idx_l=None, pos_edge=False, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            pred_score_list.append(pred_score)
            true_label_list.append(true_label)

            if stats_filename is not None:
                # positive edges
                src_agg.append(src_l_cut)
                dst_agg.append(dst_l_cut)
                ts_agg.append(ts_l_cut)
                e_idx_agg.append(e_l_cut)
                # negative edges
                src_agg.append(src_l_fake)
                dst_agg.append(dst_l_fake)
                ts_agg.append(ts_l_cut)
                e_idx_agg.append(e_l_cut)

        pred_score_list = np.concatenate(pred_score_list, axis=0)
        true_label_list = np.concatenate(true_label_list, axis=0)
        perf_dict = compute_perf_metrics(true_label_list, pred_score_list)

        if stats_filename is not None:
            src_agg = np.concatenate(src_agg, axis=0)
            dst_agg = np.concatenate(dst_agg, axis=0)
            ts_agg = np.concatenate(ts_agg, axis=0)
            e_idx_agg = np.concatenate(e_idx_agg, axis=0)
            # save to file 
            np.save(stats_filename + '_NS_src.npy', src_agg)
            np.save(stats_filename + '_NS_dst.npy', dst_agg)
            np.save(stats_filename + '_NS_ts.npy', ts_agg)
            np.save(stats_filename + '_NS_e_idx.npy', e_idx_agg)
            np.save(stats_filename + '_NS_pred_score.npy', pred_score_list)
            np.save(stats_filename + '_NS_label.npy', true_label_list)

    return perf_dict


def eval_link_pred_one_snapshot(model, snap_data, stats_filename=None, batch_size=32):
    """
    Evaluate the link prediction task
    """
    if stats_filename is not None:
        print("INFO: Test edge evaluation statistics are saved at {}".format(stats_filename))

    pos_data = snap_data['pos_e']
    neg_data = snap_data['neg_e']

    print("INFO: Number of positive edges: {}".format(len(pos_data.sources)))
    print("INFO: Number of negative edges: {}".format(len(neg_data.sources)))

    pred_score_agg, true_label_agg = [], []
    src_agg, dst_agg, ts_agg, e_idx_agg = [], [], [], []

    with torch.no_grad():
        model = model.eval()

        TEST_BATCH_SIZE = batch_size
        NUM_TEST_BATCH_POS = math.ceil(len(pos_data.sources) / TEST_BATCH_SIZE)

        NUM_TEST_BATCH_NEG = math.ceil(len(neg_data.sources) / TEST_BATCH_SIZE)
        NUM_NEG_BATCH_PER_POS_BATCH = math.ceil(NUM_TEST_BATCH_NEG / NUM_TEST_BATCH_POS)

        print("INFO: NUM_TEST_BATCH_POS:", NUM_TEST_BATCH_POS)
        print("INFO: NUM_NEG_BATCH_PER_POS_BATCH:", NUM_NEG_BATCH_PER_POS_BATCH)

        for p_b_idx in tqdm(range(NUM_TEST_BATCH_POS)):
            
            # ========== positive edges ==========
            pos_s_idx = p_b_idx * TEST_BATCH_SIZE
            pos_e_idx = min(len(pos_data.sources), pos_s_idx + TEST_BATCH_SIZE)                

            pos_src_batch = pos_data.sources[pos_s_idx: pos_e_idx]
            pos_dst_batch = pos_data.destinations[pos_s_idx: pos_e_idx]
            pos_ts_batch = pos_data.timestamps[pos_s_idx: pos_e_idx]
            pos_e_idx_batch = pos_data.edge_idxs[pos_s_idx: pos_e_idx]

            pos_prob = model.contrast_modified(pos_src_batch, pos_dst_batch, pos_ts_batch, pos_e_idx_batch, pos_edge=True, test=True)
            pos_true_label = np.ones(len(pos_src_batch))

            if stats_filename is not None:
                src_agg.append(pos_src_batch)
                dst_agg.append(pos_dst_batch)
                ts_agg.append(pos_ts_batch)
                e_idx_agg.append(pos_e_idx_batch)

                pred_score_agg.append(pos_prob.cpu().numpy())
                true_label_agg.append(pos_true_label)

            # ========== negative edges ==========
            for n_b_idx in range(NUM_NEG_BATCH_PER_POS_BATCH):
                neg_s_idx = (p_b_idx * NUM_NEG_BATCH_PER_POS_BATCH + n_b_idx) * TEST_BATCH_SIZE
                neg_e_idx = min(len(neg_data.sources), neg_s_idx + TEST_BATCH_SIZE)

                neg_src_batch = neg_data.sources[neg_s_idx: neg_e_idx]
                neg_dst_batch = neg_data.destinations[neg_s_idx: neg_e_idx]
                neg_ts_batch = neg_data.timestamps[neg_s_idx: neg_e_idx]
                neg_e_idx_batch = neg_data.edge_idxs[neg_s_idx: neg_e_idx]

                if len(neg_src_batch) > 1:
                    neg_prob = model.contrast_modified(neg_src_batch, neg_dst_batch, neg_ts_batch, neg_e_idx_batch, pos_edge=False, test=True)
                    neg_true_label = np.zeros(len(neg_src_batch))

                    if stats_filename is not None:
                        src_agg.append(neg_src_batch)
                        dst_agg.append(neg_dst_batch)
                        ts_agg.append(neg_ts_batch)
                        e_idx_agg.append(neg_e_idx_batch)

                        pred_score_agg.append(neg_prob.cpu().numpy())
                        true_label_agg.append(neg_true_label)
                else:
                    print(f"DEBUG: no negative edges in batch P-{p_b_idx}_N-{n_b_idx}!")
                    continue


    if stats_filename is not None:
        src_agg = np.concatenate(src_agg, axis=0)
        dst_agg = np.concatenate(dst_agg, axis=0)
        ts_agg = np.concatenate(ts_agg, axis=0)
        e_idx_agg = np.concatenate(e_idx_agg, axis=0)
        pred_score_agg = np.concatenate(pred_score_agg, axis=0)
        true_label_agg = np.concatenate(true_label_agg, axis=0)
        # save to file 
        np.save(stats_filename + '_src.npy', src_agg)
        np.save(stats_filename + '_dst.npy', dst_agg)
        np.save(stats_filename + '_ts.npy', ts_agg)
        np.save(stats_filename + '_e_idx.npy', e_idx_agg)
        np.save(stats_filename + '_pred_score.npy', pred_score_agg)
        np.save(stats_filename + '_label.npy', true_label_agg)




