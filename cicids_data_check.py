import pandas as pd

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

if __name__ == "__main__":
    
    #tue_data = pd.read_csv("data/tue_data.csv")
    wed_data = pd.read_csv("data/wed_data.csv")
    #print(tue_data.dtypes)
    #flow_selector_ips(tue_data, '205.174.165.73', '205.174.165.68')
    print(flow_selector_ips(wed_data, '205.174.165.73', '192.168.10.50'))
    print(flow_selector_attack(wed_data, 'DoS Hulk'))
 