"""
Training and Validation Procedure

Date:
    - Jan. 10, 2023
"""


import torch
import numpy as np
import pandas as pd
import math
import time
import logging
from eval import *


def train_val(model, train_val_data, bs, epochs, n_neighbors, criterion, optimizer, early_stopper, ngh_finder, train_val_sampler, logger, run_idx, device):
    """
    Training procedure for a TGAT model
    """
    # unpack the data, and prepare for trainiing
    train_data, val_data, new_node_val_data = train_val_data
    train_rand_sampler, val_rand_sampler, nn_val_rand_sampler = train_val_sampler
    model.ngh_finder = ngh_finder

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / bs)

    logger.info('Number of training instances: {}'.format(num_instance))
    logger.info('Number of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    for epoch in range(epochs):
        m_loss = []  # store the loss of each epoch
        np.random.shuffle(idx_list)
        epoch_start_time = time.time()
        logger.info('Start epoch {}'.format(epoch))
        for k in range(num_batch):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(s_idx + bs, num_instance - 1)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx: e_idx]
            src_l_cut, dst_l_cut = train_data.sources[batch_idx], train_data.destinations[batch_idx]
            ts_l_cut = train_data.timestamps[batch_idx]
            
            size = len(src_l_cut)
            # sample negative edges
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
            # feed in the data and learn from the error
            optimizer.zero_grad()
            model = model.train()
            pos_prob = model.contrast_modified(src_l_cut, dst_l_cut, ts_l_cut, n_neighbors)
            neg_prob = model.contrast_modified(src_l_cut, dst_l_fake, ts_l_cut, n_neighbors)

            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            m_loss.append(loss.item())

        # ==================== VALIDATION
        # validation phase use all information
        logger.info("Epoch: {}: mean loss: {}.".format(epoch, np.mean(m_loss)))

        # ===== Transductive
        val_perf_dict = eval_link_pred(model, val_rand_sampler, val_data, None, bs, n_neighbors)
        for metric_name, metric_value in val_perf_dict.items():
            logger.info('Validation statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))

        # ===== Inductive 
        nn_val_perf_dict = eval_link_pred(model, nn_val_rand_sampler, new_node_val_data, None, bs, n_neighbors)
        for metric_name, metric_value in nn_val_perf_dict.items():
            logger.info('Validation statistics: New nodes -- {}: {}'.format(metric_name, metric_value))


        if early_stopper.early_stop_check(val_perf_dict['ap']):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(run_idx, early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(run_idx, epoch))

        logger.info('Epoch {} elapsed time: {} seconds.'.format(epoch, (time.time() - epoch_start_time)))
