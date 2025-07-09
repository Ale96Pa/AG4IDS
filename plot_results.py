import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def get_cmap(n, name='Set2'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.get_cmap(name, n)


def plot_grouped_bar_chart(parameters, without_mechanism, with_mechanism,
                            box_mode = 'orange_box',
                            plot_mode = 'ids_params'):
    assert box_mode in ['orange_box', 'blue_box']
    assert plot_mode in ['ids_params', 'path_prob_params', 'feat_params_top', 'feat_params_worst', 'ag_params', 'train_perc_params']
    mechanism_name = r'$IDS \rightarrow AG$' if box_mode == 'orange_box' else r'$IDS[AG]$' if box_mode == 'blue_box' else None

    metrics = list(with_mechanism.keys())
    # Set up the bar chart
    x = np.arange(len(parameters))  # the label locations
    group_width = 0.8
    width = group_width / (len(metrics))  # the width of the bars
    # offsets = np.linspace(-width, width, num=len(metrics))  # offsets for each metric
    offsets = [(i - len(metrics) / 2) * width + width for i in range(len(metrics))]
    cmap = get_cmap(2*len(metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    mins = [1000., 1000.]
    maxs = [0., 0.]

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        x_pos = x + offsets[i]
        if with_mechanism[metric] >= without_mechanism[metric]:
            if metric in ['Accuracy', 'F1-score']:
                ax.bar(x_pos, with_mechanism[metric], width, label=f'{metric} (With {mechanism_name})', color=cmap(i*2))
                ax.bar(x_pos, without_mechanism[metric], 0.75*width, label=f'{metric} (Without {mechanism_name})', hatch='//', color=cmap(i*2+1))
                if min(with_mechanism[metric]) < mins[0]: mins[0] = min(with_mechanism[metric])
                if min(without_mechanism[metric]) < mins[0]: mins[0] = min(without_mechanism[metric])
                if max(with_mechanism[metric]) > maxs[0]: maxs[0] = max(with_mechanism[metric])
                if max(without_mechanism[metric]) > maxs[0]: maxs[0] = max(without_mechanism[metric])
            elif metric in ['FPR']:
                ax2.bar(x_pos, with_mechanism[metric], width, label=f'{metric} (With)', color=cmap(i*2))
                ax2.bar(x_pos, without_mechanism[metric], 0.75*width, label=f'{metric} (Without {mechanism_name})', hatch='//', color=cmap(i*2+1))
                if min(with_mechanism[metric]) < mins[1]: mins[1] = min(with_mechanism[metric])
                if min(without_mechanism[metric]) < mins[1]: mins[1] = min(without_mechanism[metric])
                if max(with_mechanism[metric]) > maxs[1]: maxs[1] = max(with_mechanism[metric])
                if max(without_mechanism[metric]) > maxs[1]: maxs[1] = max(without_mechanism[metric])
            else:
                raise ValueError('Metric "{}" not found during plot!'.format(metric))
        else:
            if metric in ['Accuracy', 'F1-score']:
                ax.bar(x_pos, without_mechanism[metric], width, label=f'{metric} (Without {mechanism_name})', hatch='//', color=cmap(i*2+1))
                ax.bar(x_pos, with_mechanism[metric], 0.75*width, label=f'{metric} (With {mechanism_name})', color=cmap(i*2))
                if min(with_mechanism[metric]) < mins[0]: mins[0] = min(with_mechanism[metric])
                if min(without_mechanism[metric]) < mins[0]: mins[0] = min(without_mechanism[metric])
                if max(with_mechanism[metric]) > maxs[0]: maxs[0] = max(with_mechanism[metric])
                if max(without_mechanism[metric]) > maxs[0]: maxs[0] = max(without_mechanism[metric])
            elif metric in ['FPR']:
                ax2.bar(x_pos, without_mechanism[metric], width, label=f'{metric} (Without {mechanism_name})', hatch='//', color=cmap(i*2+1))
                ax2.bar(x_pos, with_mechanism[metric], 0.75*width, label=f'{metric} (With {mechanism_name})', color=cmap(i*2))
                if min(with_mechanism[metric]) < mins[1]: mins[1] = min(with_mechanism[metric])
                if min(without_mechanism[metric]) < mins[1]: mins[1] = min(without_mechanism[metric])
                if max(with_mechanism[metric]) > maxs[1]: maxs[1] = max(with_mechanism[metric])
                if max(without_mechanism[metric]) > maxs[1]: maxs[1] = max(without_mechanism[metric])
            else:
                raise ValueError('Metric "{}" not found during plot!'.format(metric))

        # Annotate delta on top of the 'with' bars
        for j in range(len(parameters)):
            delta = with_mechanism[metric][j] - without_mechanism[metric][j]
            if delta >= 0:
                if metric in ['Accuracy', 'F1-score']:
                    ax.text(x_pos[j], with_mechanism[metric][j] + 0.005, f'+{delta:.2f}%', 
                        ha='center', va='bottom', rotation=90, fontsize=15, color='black')
                elif metric in ['FPR']:
                    ax2.text(x_pos[j], with_mechanism[metric][j] + 0.005, f'+{delta:.2f}%', 
                        ha='center', va='bottom', rotation=90, fontsize=15, color='black')
                else:
                    raise ValueError('Metric "{}" not found during plot!'.format(metric))
            else:
                if metric in ['Accuracy', 'F1-score']:
                    ax.text(x_pos[j], without_mechanism[metric][j] + 0.005, f'{delta:.2f}%', 
                        ha='center', va='bottom', rotation=90, fontsize=15, color='black')
                elif metric in ['FPR']:
                    ax2.text(x_pos[j], without_mechanism[metric][j] + 0.005, f'{delta:.2f}%', 
                        ha='center', va='bottom', rotation=90, fontsize=15, color='black')
                else:
                    raise ValueError('Metric "{}" not found during plot!'.format(metric))

    # Add labels and title
    xlabel = 'IDS Hyperparameters (depth x splits x leaves)' if plot_mode == 'ids_params' else \
            'Probability of Random Path Addition' if plot_mode == 'path_prob_params' else \
            'IDS Training Features' if plot_mode in ['feat_params_top', 'feat_params_worst'] else \
            'AG used for {}'.format(mechanism_name) if plot_mode == 'ag_params' else \
            'Percentage of Training Data (%)' if plot_mode == 'train_perc_params' else \
            None
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel('IDS Performance (Accuracy and F1-score)', fontsize=15)
    ax2.set_ylabel('IDS Performance (FPR)', fontsize=15)
    # ax.set_title('IDS Performance With and Without AG-Based Refinement')
    ax.set_xticks(x)
    # ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    if plot_mode == 'ids_params':
        ax.set_xticklabels(parameters, rotation=60, fontsize=15)
    elif plot_mode == 'path_prob_params':
        ax.set_xticklabels(parameters, fontsize=15)
    elif plot_mode == 'feat_params_top':
        ax.set_xticklabels(parameters, fontsize=15)
    elif plot_mode == 'feat_params_worst':
        ax.set_xticklabels(parameters, fontsize=15)
    elif plot_mode == 'ag_params':
        ax.set_xticklabels(parameters, fontsize=15)
    elif plot_mode == 'train_perc_params':
        ax.set_xticklabels(parameters, fontsize=15)
    else:
        raise ValueError('Something went wrong!')
    
    lgd_delta = 0.8
    if plot_mode == 'ids_params':
        ax.set_ylim(mins[0]-0.5, maxs[0]+1.3+lgd_delta)
        ax2.set_ylim(0., maxs[1]+1.3+lgd_delta)
    elif plot_mode == 'path_prob_params':
        ax.set_ylim(mins[0]-0.5, maxs[0]+0.2+lgd_delta)
        ax2.set_ylim(0., maxs[1]+0.2+lgd_delta)
    elif plot_mode == 'feat_params_top': 
        ax.set_ylim(mins[0]-0.5, maxs[0]+1.3+lgd_delta)
        ax2.set_ylim(0., maxs[1]+1.3+lgd_delta)
    elif plot_mode == 'feat_params_worst': 
        ax.set_ylim(mins[0]-0.5, maxs[0]+1.6+lgd_delta)
        ax2.set_ylim(0., maxs[1]+1.6+lgd_delta)
    elif plot_mode == 'ag_params':
        ax.set_ylim(mins[0]-0.5, maxs[0]+0.0+lgd_delta)
        ax2.set_ylim(0., maxs[1]+0.0+lgd_delta)
    elif plot_mode == 'train_perc_params':
        ax.set_ylim(mins[0]-0.5, maxs[0]+0.2+lgd_delta)
        ax2.set_ylim(0., maxs[1]+0.2+lgd_delta)
    else:
        raise ValueError('Something went wrong!')
    
    labels = ['{} ({})'.format(metrics[int(i/2)], 'With {}'.format(mechanism_name)) if i%2 == 0 else '{} ({})'.format(metrics[math.floor(i/2)], 'Without {}'.format(mechanism_name)) for i in range(2*len(metrics))]
    handles = [plt.Rectangle((0,0),1,1, color=cmap(i), hatch='x') if i%2 != 0 else plt.Rectangle((0,0),1,1, color=cmap(i)) for i, label in enumerate(labels)]
    # plt.legend(handles, labels)
    
    # Legend placing
    # # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    # # Put a legend below current axis
    # if plot_mode == 'ids_params':
    #     lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #             fancybox=True, shadow=False, ncol=len(metrics))
    # # elif plot_mode == 'ag_params':
    # #     lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.85),
    # #             fancybox=True, shadow=False, ncol=len(metrics))
    # else:
    #     lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #             fancybox=True, shadow=False, ncol=len(metrics))
    lgd = ax.legend(handles, labels, loc='upper left', fancybox=True, shadow=False, ncol=len(metrics), fontsize=10.5)

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs('plots/{}'.format(box_mode), exist_ok=True)
    plt.savefig('plots/{}/{}_bar_chart.pdf'.format(box_mode, plot_mode), bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_radar_chart(parameters, without_mechanism, with_mechanism,
                            box_mode = 'orange_box',
                            plot_mode = 'ids_params'):
    assert box_mode in ['orange_box', 'blue_box']
    assert plot_mode in ['ids_params', 'path_prob_params', 'feat_params_top', 'feat_params_worst', 'ag_params', 'train_perc_params']
    mechanism_name = r'$IDS \rightarrow AG$' if box_mode == 'orange_box' else r'$IDS[AG]$' if box_mode == 'blue_box' else None

    metrics = list(with_mechanism.keys())
    # Compute angles for radar chart
    num_params = len(parameters)
    angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    cmap = get_cmap(2*len(metrics))

    # Create subplots
    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(5 * cols, 5 * rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    mins = 1000.
    maxs = 0.

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values_with = with_mechanism[metric] + [with_mechanism[metric][0]]
        values_without = without_mechanism[metric] + [without_mechanism[metric][0]]
        if min(with_mechanism[metric]) < mins: mins = min(with_mechanism[metric])
        if min(without_mechanism[metric]) < mins: mins = min(without_mechanism[metric])
        if max(with_mechanism[metric]) > maxs: maxs = max(with_mechanism[metric])
        if max(without_mechanism[metric]) > maxs: maxs = max(without_mechanism[metric])

        ax.plot(angles, values_with, label='With {}'.format(mechanism_name), color=cmap(0))
        ax.fill(angles, values_with, color=cmap(0), alpha=0.25)

        ax.plot(angles, values_without, label='Without {}'.format(mechanism_name), color=cmap(4), linestyle='dashed')
        ax.fill(angles, values_without, color=cmap(4), alpha=0.15)

        if metric in ['Accuracy', 'F1-score']:
            ax.set_ylim(bottom = math.floor(mins), top = math.ceil(maxs))
            ax.set_yticks(np.linspace(math.floor(mins), math.ceil(maxs), num=5))

        ax.set_title(metric, size=14, y=1.1)

        # ax.set_xticks(angles[:-1])
        # ax.set_xticklabels(parameters, size=8)

        # Remove default xticks and manually add labels
        ax.set_xticks([])
        for angle, label in zip(angles[:-1], parameters):
            x = np.cos(angle)
            y = np.sin(angle)
            ha = 'left' if x > 0.1 else 'right' if x < -0.1 else 'center'
            va = 'bottom' if y > 0.1 else 'top' if y < -0.1 else 'center'
            pos = ax.get_rmax() + 0.05 if idx < 2 else ax.get_rmax() + 0.005
            ax.text(angle, pos, label, size=10, ha=ha, va=va)

        if idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.2))

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs('plots/{}'.format(box_mode), exist_ok=True)
    plt.savefig('plots/{}/{}_radar_chart.pdf'.format(box_mode, plot_mode))
    # plt.show()
    plt.close()



def extract_results_for_params(params, box_mode='orange_box', plot_mode='ids_params'):
    assert box_mode in ['orange_box', 'blue_box']
    assert plot_mode in ['ids_params', 'path_prob_params', 'feat_params_top', 'feat_params_worst', 'ag_params', 'train_perc_params']
    results_folder = 'results/{}'.format(box_mode)
    # parameter_values = ['{}x{}x{}'.format(depth, split, leaf) for depth in dt_depth for split in min_samples_split for leaf in min_samples_leaf]
    metrics = ['Accuracy', 'F1-score', 'FPR']
    without_mechanism = {met: [] for met in metrics}
    with_mechanism = {met: [] for met in metrics}
    relevant_results = [f.path for f in os.scandir(results_folder) if f.is_dir() and 
                        any('AG:type_{}_'.format(ag) in f.path for ag in params['ags']) and 
                        any('additional_path_prob_{}-'.format(ag_path_prob) in f.path for ag_path_prob in params['ag_path_probs']) and 
                        any('DATA:p_{}_'.format(train_percentage) in f.path for train_percentage in params['train_percentages']) and 
                        any('feat_{}'.format(features_mode) in f.path for features_mode in params['features_modes']) and 
                        any('IDS:depth_{}_'.format(dep) in f.path for dep in params['dt_depths']) and 
                        any('min_split_{}_'.format(sp) in f.path for sp in params['min_samples_splits']) and 
                        any('min_leaf_{}-'.format(le) in f.path for le in params['min_samples_leafs'])]
    # if plot_mode == 'feat_params_worst' and box_mode == 'orange_box':
    #     print(relevant_results)
    if plot_mode == 'ids_params':
        sorting_func = lambda x: (int(x.split('depth_')[-1].split('_')[0]), 
                                int(x.split('min_split_')[-1].split('_')[0]),
                                int(x.split('min_leaf_')[-1].split('-')[0]))
    elif plot_mode == 'path_prob_params':
        sorting_func = lambda x: float(x.split('additional_path_prob_')[-1].split('-')[0])
    elif plot_mode in ['feat_params_top', 'feat_params_worst']:
        sorting_func = lambda x: x.split('feat_')[-1].split('-')[0].split('_')[-1]
    elif plot_mode == 'ag_params':
        # order_list = ['alertNetAG', 'CiC17NetAG']
        sorting_func = lambda x: x.split('AG:type_')[-1].split('_')[0]
    elif plot_mode == 'train_perc_params':
        sorting_func = lambda x: float(x.split('DATA:p_')[-1].split('_')[0])
    else:
        raise ValueError('Something went wrong!')
    
    relevant_results = sorted(relevant_results, key=sorting_func)
    parameter_values = []
    for exp_setting in relevant_results:
        if plot_mode == 'ids_params':
            param_label = '{} x {} x {}'.format(exp_setting.split('depth_')[-1].split('_')[0],
                                                    exp_setting.split('min_split_')[-1].split('_')[0],
                                                    exp_setting.split('min_leaf_')[-1].split('-')[0],)
        elif plot_mode == 'path_prob_params':
            param_label = r'$p = {}$'.format(exp_setting.split('additional_path_prob_')[-1].split('-')[0])
        elif plot_mode in ['feat_params_top', 'feat_params_worst']:
            feat_mode_name = exp_setting.split('feat_')[-1].split('_')[0]
            if feat_mode_name in ['top', 'worst']:
                param_label = '{}-{}'.format(exp_setting.split('feat_')[-1].split('_')[0].capitalize(),
                                            exp_setting.split('feat_')[-1].split('-')[0].split('_')[-1])
            elif feat_mode_name == 'all':
                param_label = '{}'.format(exp_setting.split('feat_')[-1].split('_')[0].capitalize())
        elif plot_mode == 'ag_params':
            ag_name = exp_setting.split('AG:type_')[-1].split('_')[0]
            if ag_name == 'alertNetAG':
                param_label = 'ET'
            elif ag_name == 'CiC17NetAG':
                param_label = 'Scrape'
            elif ag_name == 'fullNetAG':
                param_label = 'ET + Scrape'
            elif ag_name == 'partialAlertNetAG':
                param_label = 'Sub(ET)'
            elif ag_name == 'partialAlertOriginalNetAG':
                param_label = 'Sub(ET) + Scrape'
            else:
                param_label = 'Mix'
        elif plot_mode == 'train_perc_params':
            param_label = '{:.0f}'.format(100.*float(exp_setting.split('DATA:p_')[-1].split('_')[0]))
        else:
            raise ValueError('Something went wrong!')
        parameter_values.append(param_label)
        file_name_template = 'unrefined_results_*.json' if box_mode == 'orange_box' else 'uninjected_results_*.json' if box_mode == 'blue_box' else None
        unrefined_results_files = glob(os.path.join(exp_setting, file_name_template))
        # print(unrefined_results_files)
        for metric in metrics:
            sum_met = 0
            counter = 0
            for file in unrefined_results_files:
                # print(file)
                with open(file) as json_data:
                    results_json = json.load(json_data)
                met = results_json['accuracy'] if metric == 'Accuracy' else results_json['weighted_f1'] if metric == 'F1-score' else results_json['fpr'] if metric == 'FPR' else None
                if met is None: raise ValueError('Something went wrong with the extraction of results!')
                sum_met += met
                counter += 1
            avg_met = sum_met/counter
            without_mechanism[metric].append(avg_met)
        file_name_template = 'refined_results_*.json' if box_mode == 'orange_box' else 'injected_results_*.json' if box_mode == 'blue_box' else None
        refined_results_files = glob(os.path.join(exp_setting, file_name_template))
        # print(refined_results_files)
        for metric in metrics:
            sum_met = 0
            counter = 0
            for file in refined_results_files:
                # print(file)
                with open(file) as json_data:
                    results_json = json.load(json_data)
                met = results_json['accuracy'] if metric == 'Accuracy' else results_json['weighted_f1'] if metric == 'F1-score' else results_json['fpr'] if metric == 'FPR' else None
                if met is None: raise ValueError('Something went wrong with the extraction of results!')
                sum_met += met
                counter += 1
            avg_met = sum_met/counter
            with_mechanism[metric].append(avg_met)
    return parameter_values, without_mechanism, with_mechanism


def plot_for_mode_and_params(plot_mode, params, box_mode='orange_box'):
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode=plot_mode)
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode=plot_mode)
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode=plot_mode)


def plot_all():
    box_modes = ['orange_box', 'blue_box']
    all_plots_params = {'orange_box': {
                                        'ids_params': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [5, 20],
                                                        'min_samples_splits': [2, 10],
                                                        'min_samples_leafs': [1, 10],},
                                        'path_prob_params': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0, 0.001, 0.01, 0.05, 0.1, 0.2],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'feat_params_top': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_10', 'top_15', 'top_20', 'top_40', 'all'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'feat_params_worst': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'ag_params': 
                                                        {'ags': ['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'train_perc_params': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                    },
                        'blue_box': {
                                        'path_prob_params': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0, 0.001, 0.01, 0.05, 0.1, 0.2],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'feat_params_top': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_10', 'top_15', 'top_20', 'top_40', 'all'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'feat_params_worst': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'ag_params': 
                                                        {'ags': ['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.6],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                        'train_perc_params': 
                                                        {'ags': ['alertNetAG'],
                                                        'ag_path_probs': [0.0],
                                                        'train_percentages': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                        'features_modes': ['top_20'],
                                                        'dt_depths': [20],
                                                        'min_samples_splits': [2],
                                                        'min_samples_leafs': [1],},
                                    }
                        }
    for box_mode in box_modes:
        all_mode_params = all_plots_params[box_mode]
        for plot_mode, params in all_mode_params.items():
            plot_for_mode_and_params(plot_mode=plot_mode,
                                    params=params,
                                    box_mode=box_mode)


def plot_for_dt_params(box_mode='orange_box'):
    params = {'ags': ['alertNetAG'],
            'ag_path_probs': [0.0],
            'train_percentages': [0.2],
            'features_modes': ['top_10'],
            'dt_depths': [5, 20],
            'min_samples_splits': [2, 10],
            'min_samples_leafs': [1, 10],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='ids_params')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='ids_params')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='ids_params')


def plot_for_feat_params_top(box_mode='orange_box'):
    params = {'ags': ['alertNetAG'],
            'ag_path_probs': [0.0],
            'train_percentages': [0.2],
            'features_modes': ['top_10', 'top_15', 'top_20', 'top_40', 'all'],
            'dt_depths': [20],
            'min_samples_splits': [2],
            'min_samples_leafs': [1],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='feat_params_top')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='feat_params_top')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='feat_params_top')
    

def plot_for_feat_params_worst(box_mode='orange_box'):
    params = {'ags': ['alertNetAG'],
            'ag_path_probs': [0.0],
            'train_percentages': [0.2],
            'features_modes': ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all'],
            'dt_depths': [20],
            'min_samples_splits': [2],
            'min_samples_leafs': [1],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='feat_params_worst')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='feat_params_worst')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='feat_params_worst')


def plot_for_train_perc_params(box_mode='orange_box'):
    params = {'ags': ['alertNetAG'],
            'ag_path_probs': [0.0],
            'train_percentages': [0.2, 0.4, 0.6, 0.8],
            'features_modes': ['all'],
            'dt_depths': [20],
            'min_samples_splits': [2],
            'min_samples_leafs': [1],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='train_perc_params')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='train_perc_params')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='train_perc_params')


def plot_for_path_prob_params(box_mode='orange_box'):
    params = {'ags': ['alertNetAG'],
            'ag_path_probs': [0.0, 0.001, 0.01, 0.05, 0.1],
            'train_percentages': [0.6],
            'features_modes': ['all'],
            'dt_depths': [20],
            'min_samples_splits': [2],
            'min_samples_leafs': [1],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='path_prob_params')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='path_prob_params')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='path_prob_params')


def plot_for_ag_params(box_mode='orange_box'):
    params = {'ags': ['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG'],
            'ag_path_probs': [0.0],
            'train_percentages': [0.6],
            'features_modes': ['all'],
            'dt_depths': [20],
            'min_samples_splits': [2],
            'min_samples_leafs': [1],}
    parameter_values, without_mechanism, with_mechanism = extract_results_for_params(params=params, box_mode=box_mode, plot_mode='ag_params')
    plot_grouped_bar_chart(parameters=parameter_values,
                        without_mechanism=without_mechanism,
                        with_mechanism=with_mechanism,
                        box_mode=box_mode,
                        plot_mode='ag_params')
    plot_radar_chart(parameters=parameter_values,
                    without_mechanism=without_mechanism,
                    with_mechanism=with_mechanism,
                    box_mode=box_mode,
                    plot_mode='ag_params')


if __name__ == '__main__':
    # plot_for_dt_params()
    plot_all()


    # # Sample data
    # parameter_values = ['Param1', 'Param2', 'Param3', 'Param4']
    # # Performance values without and with the new mechanism
    # without_mechanism = {
    #     'Acc': [75, 78, 80, 82],
    #     'F1': [70, 72, 74, 76],
    #     'FPR': [68, 70, 73, 75],
    #     'Precision': [68, 70, 73, 75],
    #     # 'Recall': [70, 72, 74, 76],
    # }
    # with_mechanism = {
    #     'Acc': [80, 83, 85, 87],
    #     'F1': [76, 78, 80, 82],
    #     'FPR': [74, 76, 78, 80],
    #     'Precision': [68, 70, 73, 75],
    #     # 'Recall': [76, 78, 80, 82],
    # }
    # plot_grouped_bar_chart(parameters=parameter_values,
    #                         without_mechanism=without_mechanism,
    #                         with_mechanism=with_mechanism)