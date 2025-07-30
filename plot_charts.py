import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


COLORS=['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc',
        '#e5d8bd','#fddaec','#f2f2f2', "#1919194F","#00ff914e","#a6ff009c",
        "#76fff6c3",]


KEYWORD_MATCH = {
                "Unspecified": r"$\star$",
                "Cloud computing": r"$\triangleleft$",
                "Cyber-Physical Systems": "CPS",
                "Smart Grids": "SG",
                "Internet of Things": r"$\triangleright$",
                "AMI System": r"$\natural$",
                "Software Defined Networking": r"$\flat$",
                "Smart Cities": "SC",
                "Enterprise network system": r"$\sharp$",
                "Smart home system": r"$\blacksquare$",
                "Industrial Control Systems": "ICS",
                "SOCs": r"$\amalg$",
                "Custom": "Custom",
                "Logic": "Logic",
                "Topologic": "Topologic",
                "Bayesian": "Bayes",
                "Scenario": "Scenario",
                "None": "None",
                "Neural Network": "NN",
                "Bayesian Network": "BN",
                "Markov Chain": "MC",
                "Artificial Immune System": "AIS",
                "Signature not ML": "Signature",
                "Anomaly and ML": "An+ML",
                "Agnostic": "Ag",
                "Hybrid": r"$\unlhd$",
                "Signature and ML": "S+ML",
                "Anomaly not ML": "An",
                "Network": "NIDS",
                "Host": "HIDS",
                "DDoS": "DDoS",
                "Multi-step attacks": "Multi",
                "Remote Code Execution": r"$\dagger$",
                "DoS": "DoS",
                "U2R": r"$\ddagger$",
                "R2L": r"$\unrhd$",
                "Key Loggers": r"$\ast$",
                "OS scan": r"$\bullet$",
                "Probing": r"$\circ$",
                "Port scan": r"$\diamond$",
                "SSH Brute Force": r"$\clubsuit$",
                "DARPA2000": "DARPA",
                "Simulation": "Sim",
                "Defcon CTF'17": r"$\spadesuit$",
                "CSE-CIC-IDS-2018": r"$\heartsuit$",
                "ISCXIDS2012": r"$\bowtie$",
                "NLS-KDD": r"$\bigstar$",
                "CTU-13": r"$\triangle$",
                "CICIoT2023": r"$\blacktriangle$",
                "CPTC-2018": r"$\triangledown$",
                "Alert Correlation": "Alert Corr.",
                "Vulnerability Detection": "Vuln. Det.",
                "Response": "Response",
                "Detection Refinement": "Detect. Refin.",
                "IDS Optimization": "IDS Optim.",
                "Runtime Detection": "Runtime Det.",
                }


def ag_generation_data():
    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 29,
                                    "Vulnerability Detection": 23,
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 25,
                                "Anomaly and ML": 10,
                                "Agnostic": 13,
                                "Hybrid": 1,
                                "Signature and ML": 3,
                                "Anomaly not ML": 0
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 11,
                            "Host": 4,
                            "Agnostic": 37
                        },
        "AG (RSQ3)": 
                    {
                        "Custom": 7,
                        "Logic": 15,
                        "Topologic": 12,
                        "Bayesian": 6,
                        "Scenario": 12
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 16,
                                "None": 0,
                                "Multi-step attacks": 33,
                                "Remote Code Execution": 0,
                                "DoS": 20,
                                "Unspecified": 1,
                                "U2R": 2,
                                "R2L": 2,
                                "Key Loggers": 0,
                                "OS scan": 0,
                                "Probing": 1,
                                "Port scan": 1,
                                "SSH Brute Force": 1
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 17,
                                "Custom": 9,
                                "Simulation": 13,
                                "Defcon CTF'17": 1,
                                "CSE-CIC-IDS-2018": 3,
                                "ISCXIDS2012": 2,
                                "NLS-KDD": 1,
                                "CTU-13": 1,
                                "CICIoT2023": 1,
                                "CPTC-2018": 1,
                                "Unspecified": 6
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 25,
                                    "Cloud computing": 1,
                                    "Cyber-Physical Systems": 6,
                                    "Smart Grids": 7,
                                    "Internet of Things": 5,
                                    "AMI System": 0,
                                    "Software Defined Networking": 1,
                                    "Smart Cities": 2,
                                    "Enterprise network system": 1,
                                    "Smart home system": 1,
                                    "Industrial Control Systems": 2,
                                    "SOCs": 1
                                },
        "ML (RSQ6)": 
                    {
                        "None": 30,
                        "Neural Network": 9,
                        "Bayesian Network": 3,
                        "Markov Chain": 8,
                        "Artificial Immune System": 2
                    },
    }

    return occurrencies


def post_ag_data():
    
    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 13,
                                    "Response": 6,
                                    "Detection Refinement": 5
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 10,
                                "Anomaly and ML": 7,
                                "Agnostic": 5,
                                "Hybrid": 0,
                                "Signature and ML": 2,
                                "Anomaly not ML": 0
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 21,
                            "Host": 2,
                            "Agnostic": 1
                        },
        "AG (RSQ3)": 
                    {
                        "Custom": 9,
                        "Logic": 8,
                        "Topologic": 4,
                        "Bayesian": 3,
                        "Scenario": 0
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 9,
                                "None": 5,
                                "Multi-step attacks": 7,
                                "Remote Code Execution": 1,
                                "DoS": 11,
                                "Unspecified": 1,
                                "U2R": 1,
                                "R2L": 1,
                                "Key Loggers": 1,
                                "OS scan": 1,
                                "Probing": 1,
                                "Port scan": 0,
                                "SSH Brute Force": 0
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 6,
                                "Custom": 9,
                                "Simulation": 3,
                                "Defcon CTF'17": 1,
                                "CSE-CIC-IDS-2018": 1,
                                "ISCXIDS2012": 1,
                                "NLS-KDD": 0,
                                "CTU-13": 0,
                                "CICIoT2023": 0,
                                "CPTC-2018": 0,
                                "Unspecified": 4
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 14,
                                    "Cloud computing": 2,
                                    "Cyber-Physical Systems": 3,
                                    "Smart Grids": 3,
                                    "Internet of Things": 0,
                                    "AMI System": 1,
                                    "Software Defined Networking": 1,
                                    "Smart Cities": 0,
                                    "Enterprise network system": 0,
                                    "Smart home system": 0,
                                    "Industrial Control Systems": 0,
                                    "SOCs": 0
                                },
        "ML (RSQ6)": 
                    {
                        "None": 15,
                        "Neural Network": 4,
                        "Bayesian Network": 2,
                        "Markov Chain": 1,
                        "Artificial Immune System": 2
                    },
    }

    return occurrencies


def ids_integrated_ag_data():

    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 13,
                                    "IDS Optimization": 6,
                                    "Runtime Detection": 6,
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 11,
                                "Anomaly and ML": 7,
                                "Agnostic": 0,
                                "Hybrid": 3,
                                "Signature and ML": 3,
                                "Anomaly not ML": 1
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 23,
                            "Host": 2,
                            "Agnostic": 0
                        },
        "AG (RSQ3)": 
                    {
                        "Custom": 10,
                        "Logic": 10,
                        "Topologic": 3,
                        "Bayesian": 2,
                        "Scenario": 0
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 11,
                                "None": 0,
                                "Multi-step attacks": 0,
                                "Remote Code Execution": 1,
                                "DoS": 17,
                                "Unspecified": 9,
                                "U2R": 3,
                                "R2L": 3,
                                "Key Loggers": 1,
                                "OS scan": 1,
                                "Probing": 0,
                                "Port scan": 1,
                                "SSH Brute Force": 1
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 9,
                                "Custom": 7,
                                "Simulation": 7,
                                "Defcon CTF'17": 1,
                                "CSE-CIC-IDS-2018": 2,
                                "ISCXIDS2012": 0,
                                "NLS-KDD": 0,
                                "CTU-13": 0,
                                "CICIoT2023": 1,
                                "CPTC-2018": 0,
                                "Unspecified": 0
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 13,
                                    "Cloud computing": 1,
                                    "Cyber-Physical Systems": 0,
                                    "Smart Grids": 6,
                                    "Internet of Things": 3,
                                    "AMI System": 1,
                                    "Software Defined Networking": 1,
                                    "Smart Cities": 0,
                                    "Enterprise network system": 0,
                                    "Smart home system": 0,
                                    "Industrial Control Systems": 0,
                                    "SOCs": 0
                                },
        "ML (RSQ6)": 
                    {
                        "None": 12,
                        "Neural Network": 4,
                        "Bayesian Network": 7,
                        "Markov Chain": 1,
                        "Artificial Immune System": 1
                    },
    }

    return occurrencies


def plot(category):
    if category == 'ag_gen':
        occurrencies = ag_generation_data()
    elif category == 'post_ag':
        occurrencies = post_ag_data()
    elif category == 'ids_integrated_ag':
        occurrencies = ids_integrated_ag_data()
    else:
        raise ValueError('Could not handle category "{}"!'.format(category))

    rsqs = list(occurrencies.keys())

    percentages = {}
    max_num_of_classes = 0
    for rsq, count_dict in occurrencies.items():
        total = sum(list(count_dict.values()))
        if len(list(count_dict.values())) > max_num_of_classes:
            max_num_of_classes = len(list(count_dict.values()))
        percentages[rsq] = {key: value/total*100 for key, value in count_dict.items()}
    # print(percentages)

    data = np.zeros((len(rsqs), max_num_of_classes))
    for rsq, count_dict in percentages.items():
        row_id = rsqs.index(rsq)
        for i, value in enumerate(count_dict.values()):
            data[row_id, i] = value
    # print(data)

    data_cum = data.cumsum(axis=1)
    # print(data_cum)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 12})
    bottom = np.zeros(8)

    for i in range(max_num_of_classes):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(rsqs, widths, height=0.5, color=COLORS[i], edgecolor="black", left=starts)

    labels = [[KEYWORD_MATCH[cls] if percentages[rsq][cls] > 0 else "" for cls in percentages[rsq]] for rsq in rsqs]
    for label_list in labels:
        label_list += [''] * (max_num_of_classes - len(label_list))
    labels = [list(row) for row in zip(*labels)]
    # print(labels)

    i=0
    for c in ax.containers:
        ax.bar_label(c, labels=labels[i], label_type='center', padding=0, fontsize=11)
        i+=1

    # plt.ylim(0,NUM_POST_AG+10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    # plt.rc('axes', labelsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    plt.gca().invert_yaxis()
    plt.xlabel("Percentage of papers per category", fontsize=15)
    plt.tight_layout()
    # plt.show()
    directory = 'plots/visualization'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, '{}.pdf'.format(category)), dpi=300)



if __name__ == "__main__":
    for category in ['ag_gen', 'post_ag', 'ids_integrated_ag']:
        plot(category)