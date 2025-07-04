import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from settings import gather_settings
from ag_utils import find_node_from_ip, import_ag
from ids_utils import read_data, define_dt_ids, dt_trainer, dt_tester, print_performance
import networkx as nx
import math
import os
import json
import random



def clean_data_with_scenarios(scenario, full_data, args):

    '''
    This function prepares the data based on the selected scenario.
    '''

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
    label_encoding_dict = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Label encoding dictionary: \n {label_encoding_dict}")

    # Shuffle dataframe and create Flow ID dictionary for test data
    selected_data = selected_data.sample(frac=1).reset_index(drop=True)
    num_train_data = math.floor(args.train_percentage*len(selected_data))
    # train_data = selected_data.iloc[:num_train_data]
    test_data = selected_data.iloc[num_train_data:]
    # test_data = test_data.reset_index(drop=True)
    flow_id_dict = dict(zip(selected_data.index, test_data['Flow ID']))
    # print(f"Flow ID encoding dictionary: \n {flow_id_dict}")

    # Drop non numeric values
    selected_data = selected_data.select_dtypes(exclude=['object'])

    # Normalize train data
    selected_data_features = selected_data.drop('Numeric Label', axis=1)

    selected_data_labels = selected_data['Numeric Label']

    scaler = MinMaxScaler()
    scaler.fit(selected_data_features)
    selected_data_features = pd.DataFrame(scaler.transform(selected_data_features), columns=selected_data_features.columns)

    if 'top' in args.features_mode:
        num_selected_features = int(args.features_mode.split('_')[-1])
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
    elif args.features_mode == 'all':
        print('Using all features...')
        selected_data_filtered_features = selected_data_features
    else:
        raise ValueError('Feature mode "{}" not available!'.format(args.features_mode))
    
    dataset = selected_data_filtered_features.join(selected_data_labels)
    train_data = dataset.iloc[:num_train_data]
    test_data = dataset.iloc[num_train_data:]
    test_data = test_data.reset_index(drop=True)

    return label_encoding_dict, flow_id_dict, train_data, test_data


def dt_refiner(dt, test_data, test_flow_dict, attack_graph):
    X_test = test_data.drop('Numeric Label', axis = 1)
    y_test = test_data['Numeric Label']
    y_pred = dt.predict(X_test)
    refined_preds = refine_predictions(y_pred, flow_map=test_flow_dict, attack_graph=attack_graph)
    scores = print_performance(y_true=y_test, y_pred=refined_preds)
    return scores


def refine_predictions(y_pred, flow_map, attack_graph):
    verbose = False
    tot_predictions = len(y_pred)
    refined_preds = []
    flow_mapping = list(flow_map.values())
    for index, pred in enumerate(y_pred):
        if not verbose: print('({}/{}) Refining predictions. This may take a while...'.format(index+1, tot_predictions), end='\r')
        if verbose: print('({}/{}) Refining predictions.'.format(index+1, tot_predictions))
        if pred != 0:
            flow = flow_mapping[index]
            source_ip = flow.split('-')[0]
            destination_ip = flow.split('-')[1]
            source_node = find_node_from_ip(attack_graph, source_ip, verbose=verbose)
            destination_node = find_node_from_ip(attack_graph, destination_ip, verbose=verbose)
            if source_node is None or destination_node is None or not nx.has_path(attack_graph, source_node, destination_node):
                if verbose: print('No attack path found from "{}" to "{}"! Refining the prediction!'.format(source_ip,destination_ip))
                refined_preds.append(0)
            else:
                if verbose: print('Found attack path from "{}" to "{}"'.format(source_ip,destination_ip))
                refined_preds.append(pred)
                
        else:
            refined_preds.append(pred)
    print('')
    return refined_preds


def run_refiner(args):
    seed = random.randint(1, 10000)
    np.random.seed(seed)
    random.seed(seed)
    experiment_class_name = 'orange_box'
    experiment_name = 'AG:type_{}_additional_path_prob_{}-IDS:depth_{}_min_split_{}_min_leaf_{}-DATA:p_{}_feat_{}'.format(args.ag,
                                                                                                                        args.ag_path_prob,
                                                                                                                        args.dt_depth,
                                                                                                                        args.min_samples_split,
                                                                                                                        args.min_samples_leaf,
                                                                                                                        args.train_percentage,
                                                                                                                        args.features_mode)
    full_data = read_data(args)
    label_dictionary, flow_dictionary, scenario_train_data, scenario_test_data = clean_data_with_scenarios("benign_dos_ftp", full_data, args)
    dt_ids = define_dt_ids(args)
    dt_ids = dt_trainer(scenario_train_data, dt_ids)
    unrefined_scores = dt_tester(dt_ids, scenario_test_data)
    ag = import_ag(args=args)
    refined_scores = dt_refiner(dt_ids, scenario_test_data, flow_dictionary, ag)
    results_folder = os.path.join(args.results_folder, experiment_class_name, experiment_name)
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, 'unrefined_results_seed_{}.json'.format(seed)), 'w') as file:
        json.dump(unrefined_scores, file)
    with open(os.path.join(results_folder, 'refined_results_seed_{}.json'.format(seed)), 'w') as file:
        json.dump(refined_scores, file)



if __name__ == "__main__":
    args = gather_settings()
    run_refiner(args)