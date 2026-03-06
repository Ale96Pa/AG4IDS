import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


COLORS=['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc',
        '#e5d8bd','#fddaec','#f2f2f2', "#1919194F","#00ff914e","#a6ff009c",
        "#76fff6c3",'#f2f2f2', "#1919194F","#00ff914e","#a6ff009c",
        "#76fff6c3",]


KEYWORD_MATCH = {
                # Integration (RSQ1)
                "Alert Correlation": "Alert Corr.",
                "Vulnerability Analysis": "Vuln. An.",
                "Response": "Resp.",
                "Detection Refinement": "Det. Ref.",
                "IDS Optimization": "IDS Optim.",
                "Runtime Detection": "Run. Det.",
                
                # IDS detection (RSQ2)
                "Signature not ML": "Sig. not ML",
                "Anomaly and ML": "An+ML",
                "Agnostic": "Ag",
                # "Hybrid": r"$\unlhd$",
                "Signature and ML": r"$\ast$", # "SML",
                "Anomaly not ML": "An",
                
                # IDS scale (RSQ2)
                "Network": "NI",
                "Host": "HI",
                
                # AGs (RSQ3)
                "Host-based": "HB",
                "State-based": "SB",
                "Vulnerability-based": "VB",
                "Attack scenario-based": "AB",
                # "Custom": "Custom",
                # "Logic": "Logic",
                # "Topologic": "Topologic",
                # "Bayesian": "Bayes",
                # "Scenario": "Scenario",
                
                # Attacks (RSQ4)
                "DDoS": "DDoS",
                "Multi-step attacks": r"$\rtimes$",
                "Remote Code Execution": r"$\dagger$",
                "DoS": "DoS",
                "U2R": r"$\ddagger$",
                "R2L": r"$\unrhd$",
                "Key Loggers": "K", # r"$\ast$"
                "OS scan": r"$\bullet$",
                "Probing": r"$\circ$",
                "Port scan": r"$\diamond$",
                "SSH Brute Force": r"$\clubsuit$",
                
                # Dataset (RSQ4)
                "DARPA2000": "DARPA",
                "Simulation": "Sim",
                "Defcon CTF'17": r"$\spadesuit$",
                "CSE-CIC-IDS-2018": r"$\heartsuit$",
                "ISCXIDS2012": r"$\bowtie$",
                "NLS-KDD": r"$\bigstar$",
                "CTU-13": r"$\triangle$",
                "CICIoT2023": r"$\blacktriangle$",
                "CPTC-2018": r"$\triangledown$",
                "Custom": "Custom",
                
                "DARPA-CT-2019": r"$\pitchfork$", # "TBD",
                "StreamSpot": r"$\Vdash$", # "TBD",
                # "CTF23": r"$\gtrdot$", # "TBD",
                "4SICS-2015": r"$\gtrdot$", # "4S",
                "CCDC-2018": r"$\unlhd$", #r"$\ast$", # "TBD",


                # Applications (RSQ5)
                "Unspecified": "U", # r"$\pitchfork$",
                "Cloud computing": r"$\triangleleft$",
                "Cyber-Physical Systems": "CPS",
                "Smart Grids": "SG",
                "Internet of Things": r"$\triangleright$",
                "AMI System": r"$\natural$",
                "Software Defined Networking": r"$\flat$",
                "Smart Cities": "C",
                "Enterprise network system": r"$\sharp$",
                "Smart home system": r"$\blacksquare$",
                "Industrial Control Systems": "ICS",
                "SOCs": r"$\amalg$",

                "Edge computing": "E",

                # ML (RSQ6)
                "None": "None",
                "Neural Network": "NN",
                "Bayesian Network": "B",
                "Markov Chain": "MC",
                "Artificial Immune System": "AI",
                "Decision Tree": "DT", # r"$\Vdash$",
                "Support Vector Machine": "SV", # r"$\gtrdot$",
                "Probabilistic Automaton": "PA",
                }


def ag_generation_data():
    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 43,
                                    "Vulnerability Analysis": 6,
                                    "Response": 8,
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 37,
                                "Anomaly and ML": 5,
                                "Agnostic": 3,
                                "Signature and ML": 1,
                                "Anomaly not ML": 0
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 32,
                            "Host": 3,
                            "Agnostic": 11
                        },
        "AG (RSQ3)": 
                    {
                        "Attack scenario-based": 19,
                        "Vulnerability-based": 2,
                        "Host-based": 10,
                        "State-based": 15,
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 13,
                                "None": 0,
                                "Multi-step attacks": 29,
                                "Remote Code Execution": 0,
                                "DoS": 3,
                                "Unspecified": 2,
                                "U2R": 2,
                                "R2L": 2,
                                "Key Loggers": 0,
                                "OS scan": 0,
                                "Probing": 1,
                                "Port scan": 1,
                                "SSH Brute Force": 0
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 15,
                                "Custom": 9,
                                "Simulation": 11,
                                "Defcon CTF'17": 1,
                                "CSE-CIC-IDS-2018": 1,
                                "ISCXIDS2012": 1,
                                "NLS-KDD": 0,
                                "CTU-13": 0,
                                "CICIoT2023": 0,
                                "CPTC-2018": 2,
                                "Unspecified": 5,
                                "4SICS-2015": 1,
                                "CCDC-2018": 1,
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 24,
                                    "Cloud computing": 1,
                                    "Cyber-Physical Systems": 5,
                                    "Smart Grids": 6,
                                    "Internet of Things": 2,
                                    "AMI System": 0,
                                    "Software Defined Networking": 1,
                                    "Edge computing": 1,
                                    "Smart Cities": 1,
                                    "Enterprise network system": 0,
                                    "Smart home system": 1,
                                    "Industrial Control Systems": 3,
                                    "SOCs": 1,
                                    "Edge computing": 1,
                                },
        "ML (RSQ6)": 
                    {
                        "None": 31,
                        "Neural Network": 6,
                        "Bayesian Network": 1,
                        "Markov Chain": 5,
                        "Artificial Immune System": 1,
                        "Decision Tree": 0,
                        "Support Vector Machine": 0,
                        "Probabilistic Automaton": 2,
                    },
    }

    return occurrencies


def post_ag_data():
    
    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 22,
                                    "Vulnerability Analysis": 0, 
                                    "Runtime Detection": 0,
                                    "Response": 17,
                                    "Detection Refinement": 15
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 22,
                                "Anomaly and ML": 4,
                                "Agnostic": 1,
                                "Hybrid": 0,
                                "Signature and ML": 3,
                                "Anomaly not ML": 0
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 27,
                            "Host": 1,
                            "Agnostic": 2
                        },
        "AG (RSQ3)": 
                    {
                        "Attack scenario-based": 11,
                        "Vulnerability-based": 7,
                        "State-based": 9,
                        "Host-based": 3,
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 9,
                                "None": 0,
                                "Multi-step attacks": 14,
                                "Remote Code Execution": 1,
                                "DoS": 2,
                                "Unspecified": 4,
                                "U2R": 3,
                                "R2L": 3,
                                "Key Loggers": 1,
                                "OS scan": 1,
                                "Probing": 3,
                                "Port scan": 0,
                                "SSH Brute Force": 0
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 7,
                                "Custom": 12,
                                "Simulation": 10,
                                "Defcon CTF'17": 0,
                                "CSE-CIC-IDS-2018": 0,
                                "ISCXIDS2012": 0,
                                "NLS-KDD": 0,
                                "CTU-13": 0,
                                "CICIoT2023": 0,
                                "CPTC-2018": 0,
                                "Unspecified": 1
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 15,
                                    "Cloud computing": 3,
                                    "Cyber-Physical Systems": 4,
                                    "Smart Grids": 3,
                                    "Internet of Things": 0,
                                    "AMI System": 1,
                                    "Software Defined Networking": 2,
                                    "Smart Cities": 1,
                                    "Enterprise network system": 0,
                                    "Smart home system": 0,
                                    "Industrial Control Systems": 1,
                                    "SOCs": 0,
                                    "Edge computing": 0,
                                },
        "ML (RSQ6)": 
                    {
                        "None": 19,
                        "Neural Network": 5,
                        "Bayesian Network": 1,
                        "Markov Chain": 1,
                        "Artificial Immune System": 3,
                        "Decision Tree": 1,
                    },
    }

    return occurrencies


def ids_integrated_ag_data():

    occurrencies = {
        "Integration (RSQ1)": 
                                {
                                    "Alert Correlation": 12,
                                    "Detection Refinement": 8,
                                    "Runtime Detection": 9,
                                },
        "Detection (RSQ2)":
                            {
                                "Signature not ML": 12,
                                "Anomaly and ML": 6,
                                "Agnostic": 0,
                                "Hybrid": 0,
                                "Signature and ML": 1,
                                "Anomaly not ML": 0
                            },
        "IDS (RSQ2)": 
                        {
                            "Network": 17,
                            "Host": 2,
                            "Agnostic": 0
                        },
        "AG (RSQ3)": 
                    {
                        "Host-based": 6,
                        "State-based": 6,
                        "Vulnerability-based": 2,
                        "Attack scenario-based": 5,
                    },
        "Attacks (RSQ4)": 
                            {
                                "DDoS": 7,
                                "None": 0,
                                "Multi-step attacks": 6,
                                "Remote Code Execution": 1,
                                "DoS": 1,
                                "Unspecified": 5,
                                "U2R": 0,
                                "R2L": 0,
                                "Key Loggers": 1,
                                "OS scan": 1,
                                "Probing": 0,
                                "Port scan": 1,
                                "SSH Brute Force": 0
                            },
        "Dataset (RSQ4)": 
                            {
                                "DARPA2000": 5,
                                "Custom": 8,
                                "Simulation": 4,
                                "Defcon CTF'17": 0,
                                "CSE-CIC-IDS-2018": 1,
                                "ISCXIDS2012": 1,
                                "NLS-KDD": 0,
                                "CTU-13": 0,
                                "CICIoT2023": 0,
                                "CPTC-2018": 0,
                                "Unspecified": 0,
                                "DARPA-CT-2019": 0,
                                "StreamSpot": 0,
                                "CTF23": 0,
                                "4SICS-2015": 0,
                                "CCDC-2018": 0,
                            },
        "Application (RSQ5)" : 
                                {
                                    "Unspecified": 13,
                                    "Cloud computing": 0,
                                    "Cyber-Physical Systems": 0,
                                    "Smart Grids": 2,
                                    "Internet of Things": 1,
                                    "AMI System": 1,
                                    "Software Defined Networking": 1,
                                    "Smart Cities": 0,
                                    "Enterprise network system": 0,
                                    "Smart home system": 0,
                                    "Industrial Control Systems": 1,
                                    "SOCs": 0,
                                    "Edge computing": 0,
                                },
        "ML (RSQ6)": 
                    {
                        "None": 9,
                        "Neural Network": 2,
                        "Bayesian Network": 5,
                        "Markov Chain": 1,
                        "Decision Tree": 2
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
    categories = ['ag_gen', 'post_ag', 'ids_integrated_ag'] # ['ag_gen', 'post_ag', 'ids_integrated_ag']
    for category in categories:
        plot(category)