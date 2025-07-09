import json
import numpy as np
import stats
import matplotlib.pyplot as plt

def renameModel(strFile):
    if "alertNet" in strFile: return "onlyAlert"
    if "CiC17" in strFile: return "AG"
    if "fullNet" in strFile: return "IDS-based AG"
    if "partialAlertNet" in strFile: return "onlyAlert (partial)"
    else: return "IDS-based AG (partial)"

def plot_risk(metric, listNetFile):
    victims_ip={
        "Web srv 16":["192.168.10.50"],
        "Ubu srv 12":["192.168.10.51"],
        "Win7 Pro 64B":['192.168.10.9'],
        "Win8.1 64B":['192.168.10.5'],
        "WinVista 64B":['192.168.10.8'],
        "Win10pro 32B":['192.168.10.14'],
        "Win10 64B":['192.168.10.15'],
        "MAC":['192.168.10.25']
    }
        
    dictVictimRisk={}
    for netFile in listNetFile:
        pathFile = netFile.replace("networks","paths").replace(".json","Path.json")
        victims_risk = {}
        with open(pathFile) as nf: paths = json.load(nf)["paths"]
        for p in paths:
            stepsP = p["path"]
            for v in victims_ip.keys():
                for ipV in victims_ip[v]:
                    if ipV in stepsP[len(stepsP)-1]:
                        if v not in victims_risk.keys(): victims_risk[v] = [p[metric]]
                        else: victims_risk[v].append(p[metric])
        
        model = renameModel(pathFile.split("paths/")[1])
        dictVictimRisk[model] = victims_risk
    
    models={}
    victims = victims_ip.keys()
    
    for mod in dictVictimRisk.keys():
        victims_risk = dictVictimRisk[mod]
        ag_risk_values = []
        
        for v in victims:
            if v in victims_risk.keys():
                avgAGrisk = round(stats.mean(victims_risk[v]),2)
                ag_risk_values.append(avgAGrisk)
            else: ag_risk_values.append(0)
       
        models[mod] = ag_risk_values    

    x = np.arange(len(victims))  # the label locations
    num_in_group = len(dictVictimRisk.keys())
    gap = 0.25
    width = (1 - gap) / num_in_group  # the width of the bars
    # width = 0.25  # the width of the bars
    multiplier = 1.5

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in models.items():
        # offset = width * multiplier
        offset = width * multiplier - (1 - gap) / 2
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=9, rotation=90)
        multiplier += 1

    ax.set_ylabel('Avg. risk')
    ax.set_xlabel('Victims')
    ax.set_xticks(x + width, victims, rotation=90)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc='upper left', ncol=3)
    ax.set_ylim(0, 1.1)

    plt.savefig("results/risk.png")

if __name__ == "__main__":   
    originalNet = "data/networks/CiC17Net.json"
    partialAlertOriginalNet = "data/networks/partialAlertOriginalNet.json"
    fullNet = "data/networks/fullNet.json"
    plot_risk("risk",[originalNet,partialAlertOriginalNet,fullNet])