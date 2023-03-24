"""
Compute Statistics about Datasets

Date:
    - March 3, 2023
"""

import numpy as np
import pandas as pd
import time
import argparse
import sys
import os
import math
import networkx as nx
from tqdm import tqdm



def load_edgelist(dataset_filename):
    """
    read the dataset's edgelist and returen it as a data frame
    """
    edgelist_df = pd.read_csv(dataset_filename)
    data_dict = {'src': edgelist_df.u.values,
                 'dst': edgelist_df.i.values,
                 'ts': edgelist_df.ts.values}
    data_df = pd.DataFrame(data_dict)
    return data_df


def get_e_recurr_at_ts(data_df, ts):
    """
    get edge recurrence for a specific timestamp
    """
    
    e_at_this_ts = data_df.loc[data_df['ts'] == ts]
    
    # get unique edges at this timestamp
    unique_e_at_ts = {}
    for i, edge in e_at_this_ts.iterrows():
        if (edge['src'], edge['dst']) not in unique_e_at_ts:
            unique_e_at_ts[(edge['src'], edge['dst'])] = 1
    
    # history
    e_before_this_ts = data_df.loc[data_df['ts'] < ts]
    unique_e_before_ts = {}
    for i, edge in e_before_this_ts.iterrows():
        if (edge['src'], edge['dst']) not in unique_e_before_ts:
            unique_e_before_ts[(edge['src'], edge['dst'])] = 1
    
    # find historical edges at the current timestamp
    current_ts_edges = set(unique_e_at_ts.keys())
    historical_edges = set(unique_e_before_ts.keys())
    intersection = current_ts_edges & historical_edges
    num_hist_edges = len(intersection) * 1.0

    # calculate reocurrence of this timestamp
    reocurrence_value = num_hist_edges/len(unique_e_at_ts)

    return reocurrence_value



def get_edge_reocurrence_ratio(data_df, recurr_stats_filename, chunk_size=10000):
    """
    generate reocurrence per timestamps for large datasets (e.g., wikipedia, mooc, reddit)
    """
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    if not os.path.isfile(recurr_stats_filename):
        reoccurrence_list = ['Nan' for _ in range(len(unique_ts))]
        reocurr_df = pd.DataFrame(zip(unique_ts, reoccurrence_list), columns=['ts', 'reocurrence'])
        reocurr_df.to_csv(recurr_stats_filename, index=False)
        ts_range = unique_ts[: min(chunk_size, len(reocurr_df))]
    else:
        reocurr_df = pd.read_csv(recurr_stats_filename)
        available_ts = reocurr_df.loc[reocurr_df['reocurrence'] == 'Nan', 'ts'].tolist()
        ts_range = available_ts[: min(chunk_size, len(available_ts))]

    reocurrence_dict = {}
    for ts_idx, ts in enumerate(ts_range):  # for ts in tqdm(unique_ts):
        # compute reocurrence value at this ts
        reocurrence_value = get_e_recurr_at_ts(data_df, ts)

        # calculate reocurrence of this timestamp
        reocurrence_dict[ts] = reocurrence_value

    # write back to file
    new_ts, new_reocur = [], []
    for i, row in reocurr_df.iterrows():
        new_ts.append(row['ts'])
        if row['ts'] in reocurrence_dict.keys():
            new_reocur.append(reocurrence_dict[row['ts']])
        else:
            new_reocur.append(row['reocurrence'])

    new_reocurr_df = pd.DataFrame(zip(new_ts, new_reocur), columns=['ts', 'reocurrence'])
    new_reocurr_df.to_csv(recurr_stats_filename, index=False)


def main():
    """
    Call different methods to calculate datasets stats.
    """
    parser = argparse.ArgumentParser('*** Data Statistics ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for computing reocurrence per timestamps.')

    try:
        args = parser.parse_args()
        print("INFO: args:", args)
    except:
        parser.print_help()
        sys.exit(0)
    
    DATA = args.data
    CHUNK_SIZE = args.chunk_size

    print("="*100)
    main_start_time = time.time()
    print(f"INFO: ==========  DATA: {DATA} ==========")

    dataset_filename = f'./data/ml_{DATA}.csv'
    data_df = load_edgelist(dataset_filename)

    # compute reocurrence per timestamp
    recurr_stats_filename = f'./LP_stats/Data_Stats/{DATA}_reocurr_per_ts.csv'
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    no_repeat = math.ceil(len(unique_ts) / CHUNK_SIZE)
    for i in tqdm(range(no_repeat)):
        start_time = time.time()
        get_edge_reocurrence_ratio(data_df, recurr_stats_filename, CHUNK_SIZE)
        print(f"INFO: =====  DATA: {DATA} ===== Chunk: {i} ===== Elapsed Time: {time.time() - start_time}s.")
    
    
    print(f"INFO: =====  DATA: {DATA} ===== Elapsed Time: {time.time() - main_start_time}s.")
    print("="*100)


if __name__ == '__main__':
    main()