"""
Plot the results 

Date:
    - March 17, 2023
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


color_map = {'JODIE':  '#d7191c',
            'DyRep':  '#fdae61',
            'TGAT':   '#ffffbf',
            'TGN':    '#abd9e9',
            'CAWN':   '#2c7bb6',
          }

figsize = (8, 5)
font_size=30



def plot_bar_per_data_diff_eval_setting(method_names, eval_value_list, eval_mode_names, metric, figname, legend=False):
    # plot
    fig, ax1 = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    bar_width = 0.15

    legend_list = []
    for idx, value_list in enumerate(eval_value_list):
        if idx == 0:
            x = np.arange(len(eval_mode_names))
        else:
            x_prev = x
            x = [x + bar_width for x in x_prev]
        legend_list.append(plt.bar(x, value_list, width=bar_width, edgecolor='black', 
                                   label=method_names[idx], color=color_map[method_names[idx]]))

    # additional setting
    plt.xticks([r + bar_width*2 for r in range(len(eval_mode_names))], eval_mode_names, fontsize=font_size - 2)
    plt.yticks(fontsize=font_size + 2)
    plt.ylabel(f'{metric}', fontsize=font_size + 4)
    if legend:
        fig.legend(handles=legend_list,
               labels=method_names,
               # borderaxespad=0.1,
               loc='lower left',
               bbox_to_anchor=(0.07, 0.87),
               ncol=5, fancybox=True, shadow=False,
               fontsize=font_size-2,
               frameon=False
               )

    # save the figure
    plt.savefig(figname)
    plt.close()


def invoke_plot_one_dataset(stats_df, LP_MODE, EVAL_MODE_LIST, EDGE_SET, METRIC, DATA, METHODS, EVAL_DICT, figname, LEGEND):

    all_evals_list, eval_mode_names = [], []
    for EVAL_MODE in EVAL_MODE_LIST:
        eval_mode_names.append(EVAL_DICT[EVAL_MODE])
        one_eval = stats_df.loc[((stats_df['LP_MODE'] == LP_MODE) & (stats_df['EVAL_MODE'] == EVAL_MODE)
                                 & (stats_df['EDGE_SET'] == EDGE_SET) & (stats_df['METRIC'] == METRIC)), DATA].tolist()
        all_evals_list.append(one_eval)

    method_eval_list = []
    for idx_method in range(len(METHODS)):
        method_eval = [all_evals_list[eval_idx][idx_method] for eval_idx in range(len(EVAL_MODE_LIST))]
        method_eval_list.append(method_eval)       

    plot_bar_per_data_diff_eval_setting(method_names=METHODS, eval_value_list=method_eval_list, 
                                       eval_mode_names=eval_mode_names, 
                                       metric=METRIC, figname=figname, legend=LEGEND)
    

def plot_one_variation_diff_edge_sets(stats_df, LP_MODE, EVAL_MODE_LIST, EDGE_SET, EDGE_SET_DICT, METRIC, DATA, METHODS, EVAL_DICT, figname, LEGEND):
    """
    plot results for one dataset, one link prediction setting, one performance metrics
    with different selected set of edges (e.g., 'all', 'hist', 'new')
    """
    # plot setting
    fig, axs = plt.subplots(figsize=figsize, nrows=len(EDGE_SET), ncols=1, sharex='col')
    # plt.subplots_adjust(bottom=0.2, left=0.2)
    bar_width = 0.15

    # different edge set
    for e_idx, e_set in enumerate(EDGE_SET):
        all_evals_list, eval_mode_names = [], []
        for EVAL_MODE in EVAL_MODE_LIST:
            eval_mode_names.append(EVAL_DICT[EVAL_MODE])
            one_eval = stats_df.loc[((stats_df['LP_MODE'] == LP_MODE) & (stats_df['EVAL_MODE'] == EVAL_MODE)
                                    & (stats_df['EDGE_SET'] == e_set) & (stats_df['METRIC'] == METRIC)), DATA].tolist()
            all_evals_list.append(one_eval)

        method_eval_list = []
        for idx_method in range(len(METHODS)):
            method_eval = [all_evals_list[eval_idx][idx_method] for eval_idx in range(len(EVAL_MODE_LIST))]
            method_eval_list.append(method_eval)   

        # plotting
        legend_list = []
        for idx, value_list in enumerate(method_eval_list):
            if idx == 0:
                x = np.arange(len(eval_mode_names))
            else:
                x_prev = x
                x = [x + bar_width for x in x_prev]
            legend_list.append(axs[e_idx].bar(x, value_list, width=bar_width, edgecolor='black', 
                                    label=METHODS[idx], color=color_map[METHODS[idx]]))

        # additional setting
        if e_idx == len(EDGE_SET) - 1:
            axs[e_idx].set_xticks([r + bar_width*2 for r in range(len(eval_mode_names))], eval_mode_names, fontsize=font_size - 2)
        if e_idx == math.floor(len(EDGE_SET)/2):
            axs[e_idx].set_ylabel(f'{METRIC}', fontsize=font_size + 4)
        # axs[e_idx].set_yticks(fontsize=font_size + 2)
        axs[e_idx].tick_params(axis='y', labelsize=font_size)
        axs[e_idx].set_title(EDGE_SET_DICT[e_set], fontsize=font_size)

        if LEGEND:
            fig.legend(handles=legend_list,
                labels=METHODS,
                # borderaxespad=0.1,
                loc='lower left',
                bbox_to_anchor=(0.07, 0.87),
                ncol=5, fancybox=True, shadow=False,
                fontsize=font_size-2,
                frameon=False
                )

    # save the figure
    plt.savefig(figname)
    plt.close()



def main():
    """
    Plotting the results
    """
    METHODS = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN']
    EVAL_DICT = {'std_rnd': 'RND',
                'std_hist': 'HIST',
                'ego_snap': 'EXH'}
    EDGE_SET_DICT  = {'all': 'ALL Link',
                      'hist': 'Historical Link',
                      'new': 'New Link'}
    DATA_LIST = ['Wikipedia', 'Reddit', 'MOOC', 'Enron', 'UCI', 'Can. Parl.', 'US Legis.'] + ['AVG']
    LP_MODE_LIST = ['trans', 'induc']
    EDGE_SET_LIST = ['all', 'hist', 'new']
    METRIC_LIST = ['AUC', 'AP']
    ALL_EVAL_MODE_LIST = ['std_rnd', 'std_hist', 'ego_snap']
    LEGEND = False

    partial_out_path = f'./LP_stats/Plots/'
    stats_filename = f'./LP_stats/paper_tables_for_plotting.csv'
    stats_df = pd.read_csv(stats_filename)

    # # NOTE: calculate the average of among all datasets; only needed one time
    # value_cols = stats_df[DATA_LIST]
    # stats_df['AVG'] = value_cols.mean(axis=1)
    # stats_df.to_csv(stats_filename, index=False)
    
    # # # ================= one plot settings
    # EVAL_MODE_1 = 'std_rnd'
    # EVAL_MODE_2 = 'std_hist'
    # EVAL_MODE_3 = 'ego_snap'
    # EVAL_MODE_LIST = [EVAL_MODE_1, EVAL_MODE_2, EVAL_MODE_3]
    # METRIC = 'AUC'
    # LP_MODE = 'trans'
    # EDGE_SET = ['all', 'hist', 'new']
    # DATA = 'Wikipedia'

    # # ================
    # generate plot for ONLY one variation
    # eval_mode_str = ''
    # for eval_mode in EVAL_MODE_LIST:
    #     eval_mode_str += f'_{eval_mode}'
    # figname = f'{partial_out_path}/{LP_MODE}/{METRIC}/{DATA}_{EDGE_SET}_{METRIC}_{LP_MODE}{eval_mode_str}.pdf'
    # invoke_plot_one_dataset(stats_df, LP_MODE, EVAL_MODE_LIST, EDGE_SET, METRIC, DATA, METHODS, EVAL_DICT, figname, LEGEND)
    # =================

    # generate plots for all different variations
    eval_mode_str = ''
    for eval_mode in ALL_EVAL_MODE_LIST:
        eval_mode_str += f'_{eval_mode}'
        
    for DATA in DATA_LIST: 
        for EDGE_SET in EDGE_SET_LIST:
            for LP_MODE in LP_MODE_LIST:
                for METRIC in METRIC_LIST:
                    figname = f'{partial_out_path}/{LP_MODE}/{METRIC}/{DATA}_{EDGE_SET}_{METRIC}_{LP_MODE}{eval_mode_str}.pdf'
                    invoke_plot_one_dataset(stats_df, LP_MODE, ALL_EVAL_MODE_LIST, EDGE_SET, METRIC, DATA, METHODS, EVAL_DICT, figname, LEGEND)


    # # ======================
    # # generate plot for ONLY one variation
    # eval_mode_str = ''
    # for eval_mode in EVAL_MODE_LIST:
    #     eval_mode_str += f'_{eval_mode}'
    # figname = f'{partial_out_path}/{LP_MODE}/{METRIC}/{DATA}_aggEdgeSet_{METRIC}_{LP_MODE}{eval_mode_str}.png'
    # plot_one_variation_diff_edge_sets(stats_df, LP_MODE, EVAL_MODE_LIST, EDGE_SET, EDGE_SET_DICT, METRIC, DATA, METHODS, EVAL_DICT, figname, LEGEND)


    




if __name__ == '__main__':
    main()