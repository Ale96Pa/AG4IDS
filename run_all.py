import subprocess
import math


def run_all_orange_box_controlled():
    # All experiments parameters
    runs = {
            # 'ids_params': 
            #                 {'ags': ['alertNetAG'],
            #                 'ag_path_probs': [0.0],
            #                 'train_percentages': [0.6],
            #                 'features_modes': ['top_20'],
            #                 'dt_depths': [5, 20],
            #                 'min_samples_splits': [2, 10],
            #                 'min_samples_leafs': [1, 10],},
            # 'path_prob_params': 
            #                 {'ags': ['alertNetAG'],
            #                 'ag_path_probs': [0.0, 0.001, 0.01, 0.05, 0.1, 0.2],
            #                 'train_percentages': [0.6],
            #                 'features_modes': ['top_20'],
            #                 'dt_depths': [20],
            #                 'min_samples_splits': [2],
            #                 'min_samples_leafs': [1],},
            # 'feat_params_top': 
            #                 {'ags': ['alertNetAG'],
            #                 'ag_path_probs': [0.0],
            #                 'train_percentages': [0.6],
            #                 'features_modes': ['top_10', 'top_15', 'top_20', 'top_40', 'all'],
            #                 'dt_depths': [20],
            #                 'min_samples_splits': [2],
            #                 'min_samples_leafs': [1],},
            'feat_params_worst': 
                            {'ags': ['alertNetAG'],
                            'ag_path_probs': [0.0],
                            'train_percentages': [0.6],
                            'features_modes': ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all'],
                            'dt_depths': [20],
                            'min_samples_splits': [2],
                            'min_samples_leafs': [1],},
            # 'ag_params': 
            #                 {'ags': ['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG'],
            #                 'ag_path_probs': [0.0],
            #                 'train_percentages': [0.6],
            #                 'features_modes': ['top_20'],
            #                 'dt_depths': [20],
            #                 'min_samples_splits': [2],
            #                 'min_samples_leafs': [1],},
            # 'train_perc_params': 
            #                 {'ags': ['alertNetAG'],
            #                 'ag_path_probs': [0.0],
            #                 'train_percentages': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            #                 'features_modes': ['top_20'],
            #                 'dt_depths': [20],
            #                 'min_samples_splits': [2],
            #                 'min_samples_leafs': [1],},
        }
    num_runs = 10
    tot_exps = 0
    for _, exp_settings in runs.items():
        tot_exps += len(exp_settings['ags']) * len(exp_settings['ag_path_probs']) * len(exp_settings['train_percentages']) * len(exp_settings['features_modes']) * len(exp_settings['dt_depths']) * len(exp_settings['min_samples_splits']) * len(exp_settings['min_samples_leafs']) * num_runs
    counter = 1
    for exp_name, exp_settings in runs.items():
        for ag in exp_settings['ags']:
            for ag_path_prob in exp_settings['ag_path_probs']:
                for train_percentage in exp_settings['train_percentages']:
                    for features_mode in exp_settings['features_modes']:
                        for dt_depth in exp_settings['dt_depths']:
                            for min_samples_split in exp_settings['min_samples_splits']:
                                for min_samples_leaf in exp_settings['min_samples_leafs']:
                                    for _ in range(num_runs):
                                        command = 'python orange_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                        '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                        ''.format(ag, 
                                                ag_path_prob,
                                                train_percentage,
                                                features_mode,
                                                dt_depth,
                                                min_samples_split,
                                                min_samples_leaf)
                                        print('| Batch {}/{} | EXP: {} | Running commands: {}'.format(counter, tot_exps, exp_name.upper(), command))
                                        counter += 1
                                        subprocess.run([command], shell=True)


def run_all_orange_box():
    ags = ['alertNetAG', 'CiC17NetAG'] # 'alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG']
    ag_path_probs = [0.0, 0.001, 0.01, 0.05, 0.1]
    train_percentages = [0.2, 0.4, 0.6, 0.8]
    features_modes = ['top_10', 'top_20', 'top_50', 'top_80', 'all']
    # ['top_5', 'top_10', 'top_15', 'top_20', 'top_25', 'top_30', 'top_35', 'top_40',
    #                 'top_45', 'top_50', 'top_55', 'top_60', 'top_65', 'top_70', 'top_75', 'top_80', 'all']
    dt_depths = [5, 10, 20]
    min_samples_splits = [2, 4, 10]
    min_samples_leafs = [1, 5, 10]
    num_runs = 10
    tot_exps = len(ags) * len(ag_path_probs) * len(train_percentages) * len(features_modes) * len(dt_depths) * len(min_samples_splits) * len(min_samples_leafs) * num_runs
    counter = 1
    for ag in ags:
        for ag_path_prob in ag_path_probs:
            for train_percentage in train_percentages:
                for features_mode in features_modes:
                    for dt_depth in dt_depths:
                        for min_samples_split in min_samples_splits:
                            for min_samples_leaf in min_samples_leafs:
                                for _ in range(num_runs):
                                    command = 'python orange_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                    '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                    ''.format(ag, 
                                            ag_path_prob,
                                            train_percentage,
                                            features_mode,
                                            dt_depth,
                                            min_samples_split,
                                            min_samples_leaf)
                                    print('| Batch {}/{} | Running commands: {}'.format(counter, tot_exps, command))
                                    counter += 1
                                    subprocess.run([command], shell=True)


def run_all_orange_box_parallel(n_parallel_processes: int = 5):
    ags = ['alertNetAG', 'CiC17NetAG'] # ['alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG']
    ag_path_probs = [0.001, 0.01, 0.05, 0.1]
    train_percentages = [0.2, 0.4, 0.6, 0.8]
    features_modes = ['top_10', 'top_20', 'top_50', 'top_80', 'all']
    # ['top_5', 'top_10', 'top_15', 'top_20', 'top_25', 'top_30', 'top_35', 'top_40',
    #                 'top_45', 'top_50', 'top_55', 'top_60', 'top_65', 'top_70', 'top_75', 'top_80', 'all']
    dt_depths = [5, 10, 20]
    min_samples_splits = [2, 4, 10]
    min_samples_leafs = [1, 5, 10]
    num_runs = 10
    commands = []
    for ag in ags:
        for ag_path_prob in ag_path_probs:
            for train_percentage in train_percentages:
                for features_mode in features_modes:
                    for dt_depth in dt_depths:
                        for min_samples_split in min_samples_splits:
                            for min_samples_leaf in min_samples_leafs:
                                for _ in range(num_runs):
                                    command = 'python orange_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                    '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                    ''.format(ag, 
                                            ag_path_prob,
                                            train_percentage,
                                            features_mode,
                                            dt_depth,
                                            min_samples_split,
                                            min_samples_leaf)
                                    commands.append(command)
    n_commands = len(commands)
    n_batches = math.ceil(n_commands/n_parallel_processes)
    for i in range(n_batches):
        comms = commands[n_parallel_processes*i: n_parallel_processes*(i+1)]
        print('| Batch {}/{} | Running commands: {}'.format(i+1, n_batches, comms))
        procs = [subprocess.Popen(comms[j], shell=True) for j in range(len(comms))]
        for p in procs:
            p.wait()




def run_all_blue_box_controlled():
    # All experiments parameters
    runs = {
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
            'ids_params': 
                            {'ags': ['alertNetAG'],
                            'ag_path_probs': [0.0],
                            'train_percentages': [0.6],
                            'features_modes': ['top_20'],
                            'dt_depths': [5, 20],
                            'min_samples_splits': [2, 10],
                            'min_samples_leafs': [1, 10],},
        }
    num_runs = 10
    tot_exps = 0
    for _, exp_settings in runs.items():
        tot_exps += len(exp_settings['ags']) * len(exp_settings['ag_path_probs']) * len(exp_settings['train_percentages']) * len(exp_settings['features_modes']) * len(exp_settings['dt_depths']) * len(exp_settings['min_samples_splits']) * len(exp_settings['min_samples_leafs']) * num_runs
    counter = 1
    for exp_name, exp_settings in runs.items():
        for ag in exp_settings['ags']:
            for ag_path_prob in exp_settings['ag_path_probs']:
                for train_percentage in exp_settings['train_percentages']:
                    for features_mode in exp_settings['features_modes']:
                        for dt_depth in exp_settings['dt_depths']:
                            for min_samples_split in exp_settings['min_samples_splits']:
                                for min_samples_leaf in exp_settings['min_samples_leafs']:
                                    for _ in range(num_runs):
                                        command = 'python blue_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                        '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                        ''.format(ag, 
                                                ag_path_prob,
                                                train_percentage,
                                                features_mode,
                                                dt_depth,
                                                min_samples_split,
                                                min_samples_leaf)
                                        print('| Batch {}/{} | EXP: {} | Running commands: {}'.format(counter, tot_exps, exp_name.upper(), command))
                                        counter += 1
                                        subprocess.run([command], shell=True)



def run_all_blue_box():
    ags = ['alertNetAG', 'CiC17NetAG'] # 'alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG']
    ag_path_probs = [0.0, 0.001, 0.01, 0.05, 0.1]
    train_percentages = [0.6]
    features_modes = ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all']
    dt_depths = [20]
    min_samples_splits = [2]
    min_samples_leafs = [1]
    num_runs = 10
    tot_exps = len(ags) * len(ag_path_probs) * len(train_percentages) * len(features_modes) * len(dt_depths) * len(min_samples_splits) * len(min_samples_leafs) * num_runs
    counter = 1
    for ag in ags:
        for ag_path_prob in ag_path_probs:
            for train_percentage in train_percentages:
                for features_mode in features_modes:
                    for dt_depth in dt_depths:
                        for min_samples_split in min_samples_splits:
                            for min_samples_leaf in min_samples_leafs:
                                for _ in range(num_runs):
                                    command = 'python blue_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                    '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                    ''.format(ag, 
                                            ag_path_prob,
                                            train_percentage,
                                            features_mode,
                                            dt_depth,
                                            min_samples_split,
                                            min_samples_leaf)
                                    print('| Batch {}/{} | Running commands: {}'.format(counter, tot_exps, command))
                                    counter += 1
                                    subprocess.run([command], shell=True)


def run_all_blue_box_parallel(n_parallel_processes: int = 5):
    ags = ['alertNetAG', 'CiC17NetAG'] # 'alertNetAG', 'CiC17NetAG', 'fullNetAG', 'partialAlertNetAG', 'partialAlertOriginalNetAG']
    ag_path_probs = [0.0, 0.001, 0.01, 0.05, 0.1]
    train_percentages = [0.6]
    features_modes = ['worst_15', 'worst_30', 'worst_50', 'worst_65', 'all']
    dt_depths = [20]
    min_samples_splits = [2]
    min_samples_leafs = [1]
    num_runs = 10
    commands = []
    for ag in ags:
        for ag_path_prob in ag_path_probs:
            for train_percentage in train_percentages:
                for features_mode in features_modes:
                    for dt_depth in dt_depths:
                        for min_samples_split in min_samples_splits:
                            for min_samples_leaf in min_samples_leafs:
                                for _ in range(num_runs):
                                    command = 'python blue_box_runner.py --ag="{}" --ag_path_prob={} --train_percentage={} ' \
                                    '--features_mode="{}" --dt_depth={} --min_samples_split={} --min_samples_leaf={}' \
                                    ''.format(ag, 
                                            ag_path_prob,
                                            train_percentage,
                                            features_mode,
                                            dt_depth,
                                            min_samples_split,
                                            min_samples_leaf)
                                    commands.append(command)
    n_commands = len(commands)
    n_batches = math.ceil(n_commands/n_parallel_processes)
    for i in range(n_batches):
        comms = commands[n_parallel_processes*i: n_parallel_processes*(i+1)]
        print('| Batch {}/{} | Running commands: {}'.format(i+1, n_batches, comms))
        procs = [subprocess.Popen(comms[j], shell=True) for j in range(len(comms))]
        for p in procs:
            p.wait()



if __name__ == '__main__':
    # run_all_orange_box()
    # run_all_orange_box_parallel()
    # run_all_blue_box()
    # run_all_blue_box_parallel()
    run_all_orange_box_controlled()
    # run_all_blue_box_controlled()