import math, os, sys, json, random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from ids.settings import gather_settings
from ids_utils import read_data, define_dt_ids, dt_trainer, dt_tester
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from attackgraph.ag_utils import find_node_from_ip, import_ag



def preprocess_data(full_data, scenario, options):

    if scenario=="benign_dos_ftp":
        label_list = ['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'FTP-Patator']
    elif scenario=="benign_dos_ftp_scan":
        label_list = ['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'FTP-Patator', 'PortScan']
    
    print(f"BUILDING DATA FOR SCENARIO {scenario}\n")

    selected_data = full_data[full_data['Label'].isin(label_list)]
    rows, cols = selected_data.shape
    print("Selected dataset dimension:")
    print(f"The selected dataset has {rows} rows and {cols} columns with these labels {selected_data['Label'].unique()}")

    # Label encoding and dictionary
    le = LabelEncoder()
    selected_data['Numeric Label'] = le.fit_transform(selected_data['Label'])

    # Shuffle dataframe and create Flow ID dictionary for test data
    selected_data = selected_data.sample(frac=1).reset_index(drop=True)
    flows_list = selected_data['Flow ID'].to_list()

    # Drop non numeric values
    selected_data = selected_data.select_dtypes(exclude=['object'])

    # Normalize train data
    selected_data_features = selected_data.drop('Numeric Label', axis=1)

    selected_data_labels = selected_data['Numeric Label']

    scaler = MinMaxScaler()
    scaler.fit(selected_data_features)
    selected_data_features = pd.DataFrame(scaler.transform(selected_data_features), columns=selected_data_features.columns)

    if 'top' in options.features_mode:
        num_selected_features = int(options.features_mode.split('_')[-1])
        _, tot_features = selected_data_features.shape
        if num_selected_features >= tot_features:
            print('Using all features...')
            selected_data_filtered_features = selected_data_features
        else:
            print('Using the best {} features out of {} total'.format(num_selected_features, tot_features))
            # Create and fit selector
            selector = SelectKBest(f_classif, k=num_selected_features)
            selector.fit(selected_data_features, selected_data_labels)
            # Get columns to keep and create new dataframe with those only
            cols_idxs = selector.get_support(indices=True)
            selected_data_filtered_features = selected_data_features.iloc[:,cols_idxs]
            # print('Selected features are: {}'.format(selected_data_features.columns[selector.get_support()]))
    elif 'worst' in options.features_mode:
        num_selected_features = int(options.features_mode.split('_')[-1])
        _, tot_features = selected_data_features.shape
        if num_selected_features >= tot_features:
            print('Using all features...')
            selected_data_filtered_features = selected_data_features
        else:
            print('Using the worst {} features out of {} total'.format(num_selected_features, tot_features))
            # Create and fit selector
            selector = SelectKBest(f_classif, k=tot_features-num_selected_features)
            selector.fit(selected_data_features, selected_data_labels)
            # Get columns to keep and create new dataframe with those only
            cols_idxs = selector.get_support(indices=True)
            cols_idxs = [i for i in range(tot_features) if i not in cols_idxs]
            selected_data_filtered_features = selected_data_features.iloc[:,cols_idxs]
            # print('Selected features are: {}'.format(selected_data_features.columns[selector.get_support()]))
    elif options.features_mode == 'all':
        print('Using all features...')
        selected_data_filtered_features = selected_data_features
    else:
        raise ValueError('Feature mode "{}" not available!'.format(args.features_mode))
    
    dataset = selected_data_filtered_features.join(selected_data_labels)

    return dataset, flows_list


def split_train_test(dataset, options):
    num_train_data = math.floor(options.train_percentage*len(dataset))
    train_data = dataset.iloc[:num_train_data]
    test_data = dataset.iloc[num_train_data:]
    test_data = test_data.reset_index(drop=True)
    return train_data, test_data


def inject_ag_features(dataset, flows_list, options):
    verbose = False
    attack_graph = import_ag(args=options)
    flow_features = []
    for flow in flows_list:
        source_ip = flow.split('-')[0]
        destination_ip = flow.split('-')[1]
        source_node = find_node_from_ip(attack_graph, source_ip, verbose=verbose)
        destination_node = find_node_from_ip(attack_graph, destination_ip, verbose=verbose)
        if source_node is None or destination_node is None or not nx.has_path(attack_graph, source_node, destination_node):
            if verbose: print('No attack path found from "{}" to "{}"! Refining the prediction!'.format(source_ip,destination_ip))
            flow_features.append(0)
        else:
            if verbose: print('Found attack path from "{}" to "{}"'.format(source_ip,destination_ip))
            flow_features.append(1)
    dataset['Exists AG Path'] = np.array(flow_features)
    return dataset


def run_optimizer(args):
    seed = random.randint(1, 10000)
    np.random.seed(seed)
    random.seed(seed)
    experiment_class_name = 'blue_box'
    experiment_name = 'AG:type_{}_additional_path_prob_{}-IDS:depth_{}_min_split_{}_min_leaf_{}-DATA:p_{}_feat_{}'.format(args.ag,
                                                                                                                        args.ag_path_prob,
                                                                                                                        args.dt_depth,
                                                                                                                        args.min_samples_split,
                                                                                                                        args.min_samples_leaf,
                                                                                                                        args.train_percentage,
                                                                                                                        args.features_mode)
    full_data = read_data(args)
    uninjected_dataset, flows_list = preprocess_data(full_data=full_data,
                                                    scenario="benign_dos_ftp",
                                                    options=args)

    uninjected_train_data, uninjected_test_data = split_train_test(dataset=uninjected_dataset,
                                                                    options=args)
    dt_ids = define_dt_ids(args)
    dt_ids = dt_trainer(uninjected_train_data, dt_ids)
    uninjected_scores = dt_tester(dt_ids, uninjected_test_data)

    injected_dataset = inject_ag_features(dataset=uninjected_dataset,
                                        flows_list=flows_list,
                                        options=args)
    injected_train_data, injected_test_data = split_train_test(dataset=injected_dataset,
                                                                options=args)
    dt_ids = define_dt_ids(args)
    dt_ids = dt_trainer(injected_train_data, dt_ids)
    refined_scores = dt_tester(dt_ids, injected_test_data)

    results_folder = os.path.join(args.results_folder, experiment_class_name, experiment_name)
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, 'uninjected_results_seed_{}.json'.format(seed)), 'w') as file:
        json.dump(uninjected_scores, file)
    with open(os.path.join(results_folder, 'injected_results_seed_{}.json'.format(seed)), 'w') as file:
        json.dump(refined_scores, file)



if __name__ == "__main__":
    args = gather_settings()
    run_optimizer(args)