import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

NUM_AG_GEN=50 #3
NUM_AG_INTEGRATED=22 #2
NUM_POST_AG=23 #1

COLORS=['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec','#f2f2f2']

def post_ag_plot():
    categories = (
        "Integration (RSQ1)",
        "Detection (RSQ2)",
        "IDS (RSQ2)",
        "AG (RSQ3)",
        "Attacks (RSQ4)",
        "Dataset (RSQ4)",
        "Application (RSQ5)",
        "ML (RSQ6)",
    )
    levels_counts = {
        "l1": np.array([NUM_POST_AG*0.55, #{164}{360}[Alert correlation]
                        NUM_POST_AG*0.11, #anomaly{320}{360}
                        NUM_POST_AG*0.9, #network 324-0
                        NUM_POST_AG*0.41, #logical 148-0
                        NUM_POST_AG*0.26, #ddos{0}{97}
                        NUM_POST_AG*0.3, #darpa{108}{216}
                        NUM_POST_AG*0.1,
                        NUM_POST_AG*0.81 #none{0}{291}
                        ]), #smartgrid{36}{72}
        "l2": np.array([NUM_POST_AG*0.25, #{65}{164}[Response]
                        NUM_POST_AG*0.39, #agnostic{180}{320}
                        NUM_POST_AG*0.1, #host 360-324
                        NUM_POST_AG*0.14, #bayes {148}{198}
                        0, #dos-0
                        NUM_POST_AG*0.05, #cse{216}{234}
                        NUM_POST_AG*0.1,
                        NUM_POST_AG*0.08 #ais{291}{320}
                        ]), #iot{0}{36}
        "l3": np.array([NUM_POST_AG*0.2, # {0}{65}[Detection improvement]
                        NUM_POST_AG*0.5, #sig{0}{180}
                        0, 
                        NUM_POST_AG*0.1, #topologic {198}{234}
                        NUM_POST_AG*0.04, #u2l{97}{111}
                        NUM_POST_AG*0.3, #custom{0}{108}
                        NUM_POST_AG*0.08,
                        NUM_POST_AG*0.11 #nn{320}{360}
                        ]), #ami{144}{173}
        "l4": np.array([0,
                        0, #sig
                        0, 
                        NUM_POST_AG*0.35, #scenario {234}{360}
                        NUM_POST_AG*0.04, #r2l{111}{124}
                        NUM_POST_AG*0.25, #simulation{270}{360}
                        NUM_POST_AG*0.2,
                        0
                        ]), #cloud{72}{144}
        "l5": np.array([0,
                        0,
                        0, 
                        0, 
                        0, #ps-0
                        NUM_POST_AG*0.1,#CTF{234}{270}
                        NUM_POST_AG*0.52,
                        0
                        ]), #nd{173}{360}
        "l6": np.array([0,
                        0,
                        0, 
                        0, 
                        0, 
                        0, #os-0
                        0,
                        0]),
        "l7": np.array([0,
                        0,
                        0, 
                        0, 
                        0, 
                        0, #kl-0
                        0,
                        0]),
        "l8": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_POST_AG*0.23, #multi-step{111}{194}
                        0,
                        0,
                        0
                        ]),
        "l9": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_POST_AG*0.43, #nd{194}{360}
                        0,
                        0,
                        0
                        ]),
    }
    labels = [
        ["Alert corr.","Anomaly","Network", "Logic","DDoS","DARPA","Sm.Grid", "None"], #l1
        ["Response","Agnostic","Host", "Bayes","","CSE","IoT", "AIS"], #l2
        ["Detect. refine.","Signature","", "Topol.","Ur","Custom","AMI", "NN"], #l3
        ["","","", "Scenario","Rl","Simulation","Cloud", ""], #l4
        ["","","", "","","ND","ND", ""], #l5
        ["","", "","","","","", ""], #l6
        ["","", "","","","","", ""], #l7
        ["","", "","","multi","","", ""], #l8
        ["","", "","","ND","","", ""], #l9
    ]

    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 12})
    bottom = np.zeros(8)

    i=0
    for boolean, weight_count in levels_counts.items():
        p = ax.barh(categories, weight_count, width, label=boolean, color=COLORS[i], edgecolor="black", left=bottom)
        bottom += weight_count
        i+=1

    i=0
    for c in ax.containers:
        ax.bar_label(c, labels=labels[i], label_type='center', padding=1)
        i+=1

    # plt.ylim(0,NUM_POST_AG+10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.rc('axes', labelsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    plt.gca().invert_yaxis()
    plt.xlabel("# papers")
    plt.tight_layout()
    # plt.show()
    plt.savefig("post-ag", dpi=300)

def ids_integrated_ag_plot():
    categories = (
        "Integration (RSQ1)",
        "Detection (RSQ2)",
        "IDS (RSQ2)",
        "AG (RSQ3)",
        "Attacks (RSQ4)",
        "Dataset (RSQ4)",
        "Application (RSQ5)",
        "ML (RSQ6)",
    )
    levels_counts = {
        "l1": np.array([NUM_AG_INTEGRATED*0.52,#{172}{360}[Alert correlation]
                        NUM_AG_INTEGRATED*0.35, #an+ml{0}{126}
                        NUM_AG_INTEGRATED*0.9, #network
                        NUM_AG_INTEGRATED*0.5, #logic{0}{180}
                        NUM_AG_INTEGRATED*0.26, #ddos{0}{93}
                        NUM_AG_INTEGRATED*0.35,#darpa{0}{126}
                        NUM_AG_INTEGRATED*0.3,
                        NUM_AG_INTEGRATED*0.15, #nn{0}{54}
                        ]),#smart
        "l2": np.array([NUM_AG_INTEGRATED*0.25,#{0}{90}[IDS optimization]
                        NUM_AG_INTEGRATED*0.15,#hybr{126}{180}
                        NUM_AG_INTEGRATED*0.1, #host{0}{108}
                        NUM_AG_INTEGRATED*0.15, #bayes{180}{240}
                        NUM_AG_INTEGRATED*0.16,#dos{93}{151}
                        NUM_AG_INTEGRATED*0.05,#cse{126}{144}
                        NUM_AG_INTEGRATED*0.1,
                        NUM_AG_INTEGRATED*0.3,#bayesian{54}{162}
                        ]),#iot{108}{144}
        "l3": np.array([NUM_AG_INTEGRATED*0.23,#{90}{172}[Runtime AG detection]
                        NUM_AG_INTEGRATED*0.1,#sml{180}{198}
                        0, 
                        NUM_AG_INTEGRATED*0.12, #topologic{306}{360}
                        NUM_AG_INTEGRATED*0.1,#u2r{151}{186}
                        NUM_AG_INTEGRATED*0.3,#custom {144}{252}
                        NUM_AG_INTEGRATED*0.05,
                        NUM_AG_INTEGRATED*0.15,#mc{162}{180}
                        ]),#ami{144}{162}
        "l4": np.array([0,
                        NUM_AG_INTEGRATED*0.4,#sig{198}{360}
                        0, 
                        NUM_AG_INTEGRATED*0.23, #scenario{240}{306}
                        NUM_AG_INTEGRATED*0.1,#r2l{186}{220}
                        NUM_AG_INTEGRATED*0.3,#simulation{144}{252}
                        NUM_AG_INTEGRATED*0.05,
                        NUM_AG_INTEGRATED*0.05, #ais{180}{198}
                        ]),#cloud{162}{180}
        "l5": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_INTEGRATED*0.03,#ps{220}{232}
                        0,
                        NUM_AG_INTEGRATED*0.5,
                        NUM_AG_INTEGRATED*0.35, #none{198}{360}
                        ]),#nd {180}{360}
        "l6": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_INTEGRATED*0.03,#os{232}{243}
                        0,
                        0,
                        0, 
                        ]),
        "l7": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_INTEGRATED*0.03,#kl{243}{255}
                        0,
                        0,
                        0, 
                        ]),
        "l8": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_INTEGRATED*0.03,#spy{255}{266}
                        0,
                        0,
                        0, 
                        ]),
        "l9": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_INTEGRATED*0.26,#nd266}{360}
                        0,
                        0,
                        0, 
                        ]),
    }
    labels = [
        ["Alert corr.","Anomaly+ML","Network", "Logic","DDoS","DARPA","SmartGrid", "NN"], #l1
        ["IDS optim.","Hybrid","Host", "Bayes","DoS","CSE","IoT", "Bayes"], #l2
        ["Runtime detect.","SigML","", "Topol.","U2R","Custom","AMI", "MC"], #l3
        ["","Signature","", "Scenario","R2L","Simulation","Cl.", "AIS"], #l4
        ["","","", "","Ps","","ND", "None"], #l5
        ["","", "", "","Os","","",""], #l6
        ["","", "","","Kl","","",""], #l7
        ["","", "","","Sp","","",""], #l8
        ["","", "","","ND","","",""], #l9
    ]

    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 12})
    bottom = np.zeros(8)

    i=0
    for boolean, weight_count in levels_counts.items():
        p = ax.barh(categories, weight_count, width, label=boolean, color=COLORS[i], edgecolor="black", left=bottom)
        bottom += weight_count
        i+=1

    i=0
    for c in ax.containers:
        ax.bar_label(c, labels=labels[i], label_type='center', padding=1)
        i+=1

    # plt.ylim(0,NUM_AG_INTEGRATED+10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.rc('axes', labelsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    plt.gca().invert_yaxis()
    plt.xlabel("# papers")
    plt.tight_layout()
    # plt.show()
    plt.savefig("integrated", dpi=300)

def ag_generation_plot():
    categories = (
        "Integration (RSQ1)",
        "Detection (RSQ2)",
        "IDS (RSQ2)",
        "AG (RSQ3)",
        "Attacks (RSQ4)",
        "Dataset (RSQ4)",
        "Application (RSQ5)",
        "ML (RSQ6)",
    )
    levels_counts = {
        "l1": np.array([NUM_AG_GEN*0.57,#{156}{360}[Alert correlation]
                        NUM_AG_GEN*0.35, #anomaly{144}{288}
                        NUM_AG_GEN*1, #agnostic
                        NUM_AG_GEN*0.3, #logic{126}{234}
                        NUM_AG_GEN*0.26, #ddos{0}{100}
                        NUM_AG_GEN*0.28,#darpa{0}{100}
                        NUM_AG_GEN*0.1,
                        NUM_AG_GEN*0.23, #nn{0}{81}
                        ]),#smart{0}{32}
        "l2": np.array([NUM_AG_GEN*0.43,#{0}{156}[Vulnerability Prediction]
                        NUM_AG_GEN*0.42,#agnostic{0}{144}
                        0, 
                        NUM_AG_GEN*0.15, #bayes{306}{360}
                        0,#dos
                        NUM_AG_GEN*0.05,#cse{100}{118}
                        NUM_AG_GEN*0.2,
                        NUM_AG_GEN*0.1,#bayesian{144}{170}
                        ]),#iot{32}{101}
        "l3": np.array([0,
                        NUM_AG_GEN*0.1,#sml{335}{360}
                        0, 
                        NUM_AG_GEN*0.2, #topologic{234}{306}
                        0,#u2r
                        NUM_AG_GEN*0.46,#custom {195}{360}
                        NUM_AG_GEN*0.05,
                        NUM_AG_GEN*0.1,#mc{81}{144}
                        ]),#ami{144}{162}
        "l4": np.array([0,
                        NUM_AG_GEN*0.13,#sig{288}{335}
                        0, 
                        NUM_AG_GEN*0.35, #scenario{0}{126}
                        0,#r2l
                        NUM_AG_GEN*0.21,#simulation{118}{195}
                        NUM_AG_GEN*0.1,
                        NUM_AG_GEN*0.05, #ais{170}{175}
                        ]),#cloud{101}{110}
        "l5": np.array([0,
                        0,
                        0, 
                        0, 
                        0,#ps
                        0,
                        NUM_AG_GEN*0.55,
                        NUM_AG_GEN*0.52, #none{110}{360}
                        ]),#nd {180}{360}
        "l6": np.array([0,
                        0, 
                        0, 
                        0, 
                        0,
                        0,#os
                        0,
                        0,
                        ]),
        "l7": np.array([0,
                        0, 
                        0, 
                        0, 
                        0,
                        0,#kl
                        0,
                        0,
                        ]),
        "l8": np.array([0,
                        0,
                        0, 
                        0, 
                        NUM_AG_GEN*0.74,#multi{93}{360}
                        0,
                        0,
                        0, 
                        ]),
        "l9": np.array([0,
                        0, 
                        0, 
                        0, 
                        0,
                        0,#nd{266}{360}
                        0,
                        0,
                        ]),
    }
    labels = [
        ["Alert corr.","Anomaly","Agnostic", "Logic","DDoS","DARPA","Sm.Gr.", "NN"], #l1
        ["Vuln. detect.","Agnostic","", "Bayes","","CsE","IoT", "Bayes"], #l2
        ["","Sig.ML","", "Topologic","","Custom","AmI", "MC"], #l3
        ["","Sig.","", "Scenario","","Simulation","Cloud", "AIS"], #l4
        ["","","", "","","","ND", "None"], #l5
        ["","", "", "","","","",""], #l6
        ["","", "", "","","","",""], #l7
        ["","", "", "","multi","","",""], #l8
        ["","", "", "","","","",""], #l9
    ]

    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})
    bottom = np.zeros(8)

    i=0
    for boolean, weight_count in levels_counts.items():
        p = ax.barh(categories, weight_count, width, label=boolean, color=COLORS[i], edgecolor="black", left=bottom)
        bottom += weight_count
        i+=1

    i=0
    for c in ax.containers:
        ax.bar_label(c, labels=labels[i], label_type='center', padding=1)
        i+=1

    # plt.ylim(0,NUM_AG_GEN+10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.rc('axes', labelsize=18)
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    plt.xlabel("# papers")
    plt.tight_layout()
    # plt.show()
    plt.savefig("ag-gen", dpi=300)

if __name__ == "__main__":
    post_ag_plot()
    ids_integrated_ag_plot()
    ag_generation_plot()