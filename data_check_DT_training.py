import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def flow_selector_ips(df, source_ip, destination_ip):
    # if df[' Source IP'] == source_ip:
    #     print("Source IP row found!")
    # else:
    #     print("Source IP row not found!")
    print(df[' Source IP'][0])
    return df.loc[(df[' Source IP'] == source_ip) & (df[' Destination IP'] == destination_ip)]#.style.hide_index().hide_columns()

def flow_selector_attack(df, attack_class):
    # if df[' Source IP'] == source_ip:
    #     print("Source IP row found!")
    # else:
    #     print("Source IP row not found!")
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
    # rows, cols = data.shape
    # print("Full dataset dimension:")
    # print(f"The full dataset has {rows} rows and {cols} columns")

    # Check for duplicates
    duplicates =  data[data.duplicated()]
    # print(f"The number of duplicates is {len(duplicates)}")

    data.drop_duplicates(inplace=True)
    # rows, cols = data.shape
    # print("Full dataset dimension:")
    # print(f"The full dataset after dropping duplicates has {rows} rows and {cols} columns")

    # Check for missing and infinite values
    # missing_values = data.isna().sum()
    # print("Missing values: ")
    # print(missing_values.loc[missing_values > 0])
    # numeric_columns = data.select_dtypes(include=np.number).columns
    # infinite_values = np.isinf(data[numeric_columns]).sum()
    # print("Infinite values: ")
    # print(infinite_values[infinite_values > 0])

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # print("New missing values: ")
    # missing_values = data.isna().sum()
    # print(missing_values.loc[missing_values > 0])

    # Filling missing values with median
    med_flow_bytes = data['Flow Bytes/s'].median()
    med_flow_packets = data[' Flow Packets/s'].median()
    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(med_flow_bytes)
    data[' Flow Packets/s'] = data[' Flow Packets/s'].fillna(med_flow_packets)

    # print(data[' Label'].unique())
    # print(data[' Label'].value_counts())

    # Remove whitespace before column names
    data = data.rename(columns=lambda x: x.strip())
    return data 

def clean_data_with_scenarios(scenario, full_data):

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
    train_data = selected_data.iloc[:1342577]
    test_data = selected_data.iloc[1342577:]
    test_data = test_data.reset_index(drop=True)
    flow_id_dict = dict(zip(selected_data.index, test_data['Flow ID']))
    # print(f"Flow ID encoding dictionary: \n {flow_id_dict}")

    # Drop non numeric values
    train_data = train_data.select_dtypes(exclude=['object'])
    test_data = test_data.select_dtypes(exclude=['object'])

    # Normalize train data
    train_features = train_data.drop('Numeric Label', axis=1)
    test_features = test_data.drop('Numeric Label', axis=1)

    train_attacks = train_data['Numeric Label']
    test_attacks = test_data['Numeric Label']

    scaler = MinMaxScaler()
    scaler.fit(train_features)
    features = pd.DataFrame(scaler.transform(train_features), columns=train_features.columns)
    scaled_train_data = features.join(train_attacks)

    scaler.fit(test_features)
    features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)
    scaled_test_data = features.join(test_attacks)

    return label_encoding_dict, flow_id_dict, scaled_train_data, scaled_test_data

def dt_trainer(train_data, test_data, test_flow_dict):
    
    X_train = train_data.drop('Numeric Label', axis = 1)
    X_test = test_data.drop('Numeric Label', axis = 1)
    y_train = train_data['Numeric Label']
    y_test = test_data['Numeric Label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_test, random_state=0)

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"DT Accuracy: {accuracy}")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

    # Print flow ID, ground truth label and predicted label for mismatches
    with open('prediction_mismatches.txt', 'w') as f:
        for idx, val in y_test.items():
            # print(f"Index: {idx}, Value: {val}")
            if val != y_pred[idx]:
                print(f"Index is: {idx} | Ground truth is: {val} | Predicted label by DT is: {y_pred[idx]} | Corresponding FLOW ID is: {list(test_flow_dict.values())[idx]}", file=f)

    # Print flow ID, ground truth label and predicted label for attack predictions by DT
    attack_labels = [1, 2, 3, 4, 5]
    with open('prediction_of_attacks.txt', 'w') as f:
        for idx, val in enumerate(y_pred):
            # print(f"Index: {idx}, Value: {val}")
            if val in attack_labels:
                print(f"Index is: {idx} | Ground truth is: {val} | Predicted label by DT is: {y_pred[idx]} | Corresponding FLOW ID is: {list(test_flow_dict.values())[idx]}", file=f)


if __name__ == "__main__":

    mon_data = pd.read_csv("data/Monday-WorkingHours.pcap_ISCX.csv")
    tue_data = pd.read_csv("data/Tuesday-WorkingHours.pcap_ISCX.csv")
    wed_data = pd.read_csv("data/Wednesday-WorkingHours.pcap_ISCX.csv")
    fri_data = pd.read_csv("data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

    # flow_selector_ips(tue_data, '205.174.165.73', '205.174.165.68')
    # print(flow_selector_ips(wed_data, '205.174.165.73', '192.168.10.50'))
    # print(flow_selector_attack(wed_data, 'DoS Hulk'))

    full_data = unique_data_creation(mon_data, tue_data, wed_data, fri_data)
    label_dictionary, flow_dictionary, scenario_train_data, scenario_test_data = clean_data_with_scenarios("benign_dos_ftp", full_data)
    dt_trainer(scenario_train_data, scenario_test_data, flow_dictionary)

 