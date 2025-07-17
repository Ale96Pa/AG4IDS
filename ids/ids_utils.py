import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score



def define_dt_ids(args):
    dt = DecisionTreeClassifier(max_depth=args.dt_depth,
                                min_samples_split=args.min_samples_split,
                                min_samples_leaf=args.min_samples_leaf)
    return dt


def dt_trainer(train_data, dt):
    X_train = train_data.drop('Numeric Label', axis = 1)
    y_train = train_data['Numeric Label']
    dt.fit(X_train, y_train)
    return dt


def dt_tester(dt, test_data):
    X_test = test_data.drop('Numeric Label', axis = 1)
    y_test = test_data['Numeric Label']
    y_pred = dt.predict(X_test)
    scores = print_performance(y_true=y_test, y_pred=y_pred)
    return scores


def print_performance(y_true, y_pred):
    # print(f"Classification Report: \n {classification_report(y_true, y_pred)}")
    # confusion_matrix(y_true, y_pred)
    perf_scores = compute_ids_scores(y_true=y_true, y_pred=y_pred)
    print('IDS performance report:' \
    '\n\tAccuracy = {:.3f}%' \
    '\n\tWeighted Precision = {:.3f}%' \
    '\n\tWeighted Recall = {:.3f}%' \
    '\n\tWeighted F1 = {:.3f}%' \
    '\n\tFPR = {:.3f}%'.format(perf_scores['accuracy'],
                                perf_scores['weighted_precision'],
                                perf_scores['weighted_recall'],
                                perf_scores['weighted_f1'],
                                perf_scores['fpr'],))
    return perf_scores


def read_data(args):
    mon_data = pd.read_csv(os.path.join(args.csvs_folder, "Monday-WorkingHours.pcap_ISCX.csv"))
    tue_data = pd.read_csv(os.path.join(args.csvs_folder, "Tuesday-WorkingHours.pcap_ISCX.csv"))
    wed_data = pd.read_csv(os.path.join(args.csvs_folder, "Wednesday-WorkingHours.pcap_ISCX.csv"))
    fri_data = pd.read_csv(os.path.join(args.csvs_folder, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"))
    full_data = unique_data_creation(mon_data, tue_data, wed_data, fri_data)
    return full_data


def get_all_flows_from_data(args):
    mon_data = pd.read_csv(os.path.join(args.csvs_folder, "Monday-WorkingHours.pcap_ISCX.csv"))
    tue_data = pd.read_csv(os.path.join(args.csvs_folder, "Tuesday-WorkingHours.pcap_ISCX.csv"))
    wed_data = pd.read_csv(os.path.join(args.csvs_folder, "Wednesday-WorkingHours.pcap_ISCX.csv"))
    fri_data = pd.read_csv(os.path.join(args.csvs_folder, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"))
    data_list = [mon_data, tue_data, wed_data, fri_data]
    data = pd.concat(data_list)
    data.drop_duplicates(inplace=True)
    all_flows = data['Flow ID'].to_list()
    all_flows = ['{}-{}'.format(flow.split('-')[0], flow.split('-')[1]) for flow in all_flows]
    unique_flows = list(set(all_flows))
    # print('unique_flows: {}'.format(unique_flows))
    # print('len(unique_flows): {}'.format(len(unique_flows)))
    return unique_flows


def flow_selector_ips(df, source_ip, destination_ip):
    print(df[' Source IP'][0])
    return df.loc[(df[' Source IP'] == source_ip) & (df[' Destination IP'] == destination_ip)]#.style.hide_index().hide_columns()


def flow_selector_attack(df, attack_class):
    print(df[' Source IP'][0])
    return df.loc[df[' Label'] == attack_class]#.style.hide_index().hide_columns()


def unique_data_creation(mon_data, tue_data, wed_data, fri_data):
    '''
    This function creates a unique file with all the data, while checking for duplicates, infinite and missing values.
    '''
    data_list = [mon_data, tue_data, wed_data, fri_data]
    data_list_dict = {0:'Monday data', 1:'Tuesday data', 2:'Wednesday data', 3:'Friday data'}

    print("Seperate datasets dimensions: ")
    for i, data in enumerate(data_list):
        rows, cols = data.shape
        print(f"{data_list_dict.get(i)} data has {rows} rows and {cols} columns")

    data = pd.concat(data_list)
    # Check for duplicates
    data.drop_duplicates(inplace=True)
    # Check for missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Filling missing values with median
    med_flow_bytes = data['Flow Bytes/s'].median()
    med_flow_packets = data[' Flow Packets/s'].median()
    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(med_flow_bytes)
    data[' Flow Packets/s'] = data[' Flow Packets/s'].fillna(med_flow_packets)
    # Remove whitespace before column names
    data = data.rename(columns=lambda x: x.strip())
    return data


def compute_ids_scores(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tot_preds = len(y_pred)
    fps = 0
    actual_negatives = 0
    for i in range(len(y_true)):
        pred = y_pred[i]
        gt = y_true[i]
        if pred != 0 and gt == 0:
            fps += 1
        if gt == 0:
            actual_negatives += 1
    
    scores = {}
    scores['accuracy'] = accuracy_score(y_true, y_pred)*100.

    scores['micro_precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='micro')*100.
    scores['macro_precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='macro')*100.
    scores['weighted_precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')*100.

    scores['micro_recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='micro')*100.
    scores['macro_recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='macro')*100.
    scores['weighted_recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')*100.

    scores['micro_f1'] = f1_score(y_true=y_true, y_pred=y_pred, average='micro')*100.
    scores['macro_f1'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro')*100.
    scores['weighted_f1'] = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')*100.

    scores['fpr'] = (float(fps) / actual_negatives)*100.

    return scores
