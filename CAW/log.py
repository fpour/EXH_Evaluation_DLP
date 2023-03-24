import logging
import time
import sys
import os
from utils import *
from pathlib import Path


def set_up_logger(args, sys_argv):
    # set up running log
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    # runtime_id = '{}-{}-{}-{}-{}-{}-{}'.format(str(time.time()), args.data, args.mode[0], args.agg, n_layer, 'k'.join(n_degree), args.pos_dim)
    runtime_id = 'caw_{}_{}_{}_{}_{}_{}'.format(args.data, args.neg_sample, args.agg, n_layer, 'k'.join(n_degree), args.pos_dim)
    logging.basicConfig(level=logging.INFO, filemode='w', force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_path = 'log/{}.log'.format(runtime_id) #TODO: improve log name

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
    # os.mkdir(checkpoint_dir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # os.mkdir(best_model_dir)
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda run_idx, epoch: (checkpoint_dir + 'checkpoint-run-{}-epoch-{}.pth'.format(run_idx, epoch))
    get_best_model_path = lambda run_idx: (best_model_dir + '{}_{}.pth'.format(runtime_id, run_idx))
    # best_model_path = best_model_dir + 'best-model.pth'

    # return logger, get_checkpoint_path, best_model_path
    return logger, get_checkpoint_path, get_best_model_path


def save_oneline_result(dir, args, test_results):
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    with open(dir+'oneline_results.txt', 'a') as f:
        elements = [str(e) for e in [args.data, args.mode[0], args.agg, n_layer, 'k'.join(n_degree), args.pos_enc, args.pos_dim, args.walk_pool, *[str(v)[:6] for v in test_results]]]
        f.write('\t'.join(elements)+'\n')
