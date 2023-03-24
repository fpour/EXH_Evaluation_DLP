import logging
import time
import sys
import os
from pathlib import Path



def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in range(num_neighbors)]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


def set_up_logger(args, sys_argv):
    # set up running log
    # n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    # n_degree = [str(n) for n in n_degree]
    # runtime_id = 'TGAT_{}_{}_{}_{}_{}_{}'.format(args.data, args.neg_sample, args.agg_method, args.attn_mode, n_layer, 'k'.join(n_degree))
    runtime_id = 'TGAT_{}_{}_{}_{}_{}'.format(args.data, args.neg_sample, args.agg_method, args.attn_mode, args.n_layer)
    logging.basicConfig(level=logging.INFO, filemode='w', force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_path = 'log/{}.log'.format(runtime_id)

    if os.path.exists(file_path):
        print("DEBUG: Previous log file is deleted, a new one will be created.")
        os.remove(file_path)
    else:
        print("DEBUG: Log file does not exist, will be created.")

    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    # set up model parameters log
    checkpoint_root = './saved_checkpoints/'
    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_root = './best_models/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.mkdir(best_model_root)
        logger.info('Create directory'.format(best_model_root))

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda run_idx, epoch: (checkpoint_dir + 'checkpoint-run-{}-epoch-{}.pth'.format(run_idx, epoch))
    get_best_model_path = lambda run_idx: (best_model_dir + '{}_{}.pth'.format(runtime_id, run_idx))

    return logger, get_checkpoint_path, get_best_model_path