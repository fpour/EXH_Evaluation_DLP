"""
Generate statistics of dynamic graph datasets
These information are mainly needed for the 'Dataset Statistics' table of the paper

Date:
    - Jan. 24, 2023

"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import argparse
import sys
import concurrent.futures
import networkx as nx


def get_args():
    parser = argparse.ArgumentParser('*** DLP Results Interpretation ***')
    # Related to process_stats
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='wikipedia')

    try:
        args = parser.parse_args()
        print("INFO: args:", args)
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

def load_edgelist(dataset_name):
    """
    read the dataset's edgelist and returen it as a data frame
    """
    edgelist_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    data_dict = {'src': edgelist_df.u.values,
                 'dst': edgelist_df.i.values,
                 'ts': edgelist_df.ts.values}
    data_df = pd.DataFrame(data_dict)
    return data_df

def get_stats_count(data_df):
    """
    return statistics of the dataset
    """
    num_nodes = len(np.unique(np.array(data_df['src'].tolist() + data_df['dst'].tolist())))
    num_edges = data_df.shape[0]
    num_ts_unique = len(np.unique(np.array(data_df['ts'].tolist())))

    uniqu_e = {}
    for idx, row in data_df.iterrows():
        if (row['src'], row['dst']) not in uniqu_e:
            uniqu_e[(row['src'], row['dst'])] = 1

    # avg_e_repeat_per_ts, avg_e_unique_per_ts = get_avg_e_repeat_per_ts(data_df)
    # avg_num_e_per_ts = get_avg_e_per_ts(data_df)
    # avg_density = avg_num_e_per_ts / (num_nodes*(num_nodes-1)/2)

    stats_count = {'num_node': num_nodes,
                   'num_edge_tot': num_edges,
                   'num_edge_unique': len(uniqu_e),
                   'num_timestamp': num_ts_unique,
                #    'avg_num_e_per_ts': avg_num_e_per_ts,
                #    'avg_density_per_ts': avg_density,
                #    'avg_e_repeat_per_ts': avg_e_repeat_per_ts,
                #    'avg_e_unique_per_ts': avg_e_unique_per_ts,
                #    'avg_degree_per_ts': get_avg_degree(data_df),
                   'avg_hist_to_total_unique_ratio_only_last_ts_per_ts (reoccurrence local)': get_edge_reocurrence_ratio_over_one_last_ts(data_df),
                #    'avg_hist_to_total_unique_ratio_per_ts (reoccurrence global)': get_edge_reocurrence_ratio_parallel(data_df),
                #    'avg_hist_to_total_unique_ratio_per_ts': get_edge_reocurrence_ratio(data_df),
                   }
    return stats_count


def get_avg_e_per_ts(data_df):
    """
    get average number of edges at each timestamp
    """
    sum_num_e_per_ts = 0
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    for ts in unique_ts:
        num_e_at_this_ts = len(data_df.loc[data_df['ts'] == ts])
        sum_num_e_per_ts += num_e_at_this_ts
    avg_num_e_per_ts = (sum_num_e_per_ts * 1.0) / len(unique_ts)
    
    print(f"INFO: avg_num_e_per_ts: {avg_num_e_per_ts}")
    return avg_num_e_per_ts


def get_avg_e_repeat_per_ts(data_df):
    sum_e_repeat = 0
    sum_e_unique = 0
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    for ts in tqdm(unique_ts):
        e_at_this_ts = data_df.loc[data_df['ts'] == ts]
        unique_e_ts = {}
        for i, edge in e_at_this_ts.iterrows():
            if (edge['src'], edge['dst']) not in unique_e_ts:
                unique_e_ts[(edge['src'], edge['dst'])] = 1
            else:
                unique_e_ts[(edge['src'], edge['dst'])] += 1
        for k, v in unique_e_ts.items():
            if v != 1:
                sum_e_repeat += (v - 1)
            else:
                sum_e_unique += v
        
    avg_e_repeat_per_ts = (sum_e_repeat * 1.0) / len(unique_ts)
    avg_e_unique_per_ts = (sum_e_unique * 1.0) / len(unique_ts)

    print(f"INFO: avg_e_repeat_per_ts: {avg_e_repeat_per_ts}")
    print(f"INFO: avg_e_unique_per_ts: {avg_e_unique_per_ts}")

    return avg_e_repeat_per_ts, avg_e_unique_per_ts


def get_edge_reocurrence_ratio(data_df):
    reocurrence_list = []
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    for ts in unique_ts:  # for ts in tqdm(unique_ts):
        e_at_this_ts = data_df.loc[data_df['ts'] == ts]
        # tot_num_e = e_at_this_ts.shape[0]
        
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
        reocurrence_list.append(num_hist_edges/len(unique_e_at_ts))

    return np.mean(np.array(reocurrence_list))


def get_e_recurr_at_ts(ts):
    """
    get edge recurrence for a specific timestamp
    """
    # print(f"DEBUG: This is another process for timestamp: {ts}", flush=True)
    
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


def get_edge_reocurrence_ratio_parallel(data_df):
    reocurrence_list = []
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        statr_time = time.perf_counter()
        reocurrence_list = list(executor.map(get_e_recurr_at_ts, unique_ts))
        end_time = time.perf_counter()
    print(f"INFO: Edge reocurrence calculation finished in {end_time - statr_time} seconds.")

    return np.mean(np.array(reocurrence_list))


def get_avg_degree(data_df):
    """
    get average degree over the timestamps
    """
    degree_avg_at_ts_list = []
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    for ts in unique_ts:  
        e_at_this_ts = data_df.loc[data_df['ts'] == ts]
        G = nx.MultiGraph()
        for idx, e_row in e_at_this_ts.iterrows():
            G.add_edge(e_row['src'], e_row['dst'], weight=e_row['ts'])
        nodes = G.nodes()
        degrees = [G.degree[n] for n in nodes]
        degree_avg_at_ts_list.append(np.mean(degrees))

    print(f"INFO: avg_degree: {np.mean(degree_avg_at_ts_list)}")
    
    return np.mean(degree_avg_at_ts_list)


def get_edge_reocurrence_ratio_over_one_last_ts(data_df):
    reocurrence_list = [0]  # for the first timestamp
    unique_ts = np.unique(np.array(data_df['ts'].tolist()))
    for idx, ts in tqdm(enumerate(unique_ts)):  # for ts in tqdm(unique_ts):
        if idx != 0:
            e_at_this_ts = data_df.loc[data_df['ts'] == ts]
            
            # get unique edges at this timestamp
            unique_e_at_ts = {}
            for i, edge in e_at_this_ts.iterrows():
                if (edge['src'], edge['dst']) not in unique_e_at_ts:
                    unique_e_at_ts[(edge['src'], edge['dst'])] = 1
            
            # history
            e_before_this_ts = data_df.loc[data_df['ts'] == unique_ts[idx - 1]]  # only the exact previous timestamp
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
            reocurrence_list.append(num_hist_edges/len(unique_e_at_ts))

    return np.mean(np.array(reocurrence_list))



# def main():
#     """
#     Execute required functions
#     """
#     args, _ = get_args()
#     DATA = args.data
    
#     print("="*100)
#     statr_time = time.time()
#     print(f"INFO: ==========  DATA: {DATA} ==========")
#     data_df = load_edgelist(DATA)
#     stats_count = get_stats_count(data_df)
    
#     for k, v in stats_count.items():
#         print(f"\t{k}: {v}")

#     print(f"INFO: =====  DATA: {DATA} ===== Elapsed Time: {time.time() - statr_time}s.")
#     print("="*100)


# if __name__ == '__main__':
#     main()


# ====================================================
# ==== in order to make the parallel version works...

args, _ = get_args()
DATA = args.data

print("="*100)
statr_time = time.time()
print(f"INFO: ==========  DATA: {DATA} ==========")
data_df = load_edgelist(DATA)
stats_count = get_stats_count(data_df)

for k, v in stats_count.items():
    print(f"\t{k}: {v}")

print(f"INFO: =====  DATA: {DATA} ===== Elapsed Time: {time.time() - statr_time}s.")
print("="*100)