import json, math, time, csv, sys, os
import pandas as pd
import networkx as nx
import numpy as np
import stats
import matplotlib.pyplot as plt

"""
This function evaluate the likelihood and impact of a given VULNERABILITY object
@input: vulnerability object
"""
def get_risk_by_vuln(vuln):
    likelihood=1
    impact=1
    
    if "cvssMetricV2" in vuln["metrics"]:
        metricV2 = vuln["metrics"]["cvssMetricV2"][0]
        likelihood=metricV2["exploitabilityScore"]/20
        impact=metricV2["impactScore"]
    
    if ("cvssMetricV30" in vuln["metrics"] and vuln["metrics"]["cvssMetricV30"]) or ("cvssMetricV31" in vuln["metrics"] and vuln["metrics"]["cvssMetricV31"]):
        if "cvssMetricV30" in vuln["metrics"]: metricV3 = vuln["metrics"]["cvssMetricV30"][0]
        else: metricV3 = vuln["metrics"]["cvssMetricV31"][0]
        likelihood=metricV3["exploitabilityScore"]/8.22
        impact=metricV3["impactScore"]
    return likelihood,impact

"""
This function calculates the risk metrics for a given path (expressed as ordered list of vulnerabilities)
@input: vuln_ids <List of String>: List of vulnerability IDs (e.g., ["CVE-2019-24239", "CVE-2004-23423"])
@input: vulnerabilities <List of Vulnerabilities>: List of ALL vulnerabilities from json file

@Output: object containing impact, likelihood, and risk of the attack path
"""
def compute_risk_analysis(vuln_ids, vulns_list):
    impact_scores=[]
    exploit_scores=[]
    for v_curr in vuln_ids:
        for v_gt in vulns_list:
            if v_gt["id"] in v_curr:
                likelihood,impact=get_risk_by_vuln(v_gt)
                exploit_scores.append(likelihood)
                impact_scores.append(impact)
    
    lambda_exploit_scores = []
    for expl_s in exploit_scores:
        lambda_exploit_scores.append(1/expl_s)
    
    # lik_risk = 0
    # MTAO = sum(lambda_exploit_scores)
    # # MTAO = (sum(lambda_exploit_scores)-min(lambda_exploit_scores))/sum(lambda_exploit_scores)
    # if MTAO>0:
    #     argument = (MTAO-min(lambda_exploit_scores))/MTAO
    #     lik_risk = (-20)*math.log(argument,10)
            
    lik_risk = (sum(lambda_exploit_scores)-min(lambda_exploit_scores))/sum(lambda_exploit_scores) if len(lambda_exploit_scores)>0 else 0
    imp_risk = (impact_scores[len(impact_scores)-1])/max(impact_scores) if len(impact_scores)>0 and max(impact_scores)>0 else 0

    # if lik_risk>1: lik_risk=1
    # if imp_risk>1: imp_risk=1

    return {
        "impact": imp_risk,
        "likelihood": lik_risk,
        "risk":(imp_risk)*(lik_risk),
    }

"""
These functions checks the pre-post condition chaining
"""
def get_req_privilege(str_priv):
    if str_priv == "NONE" or str_priv == "LOW":
        return "guest"
    elif str_priv == "SINGLE" or str_priv == "MEDIUM":
        return "user"
    else:
        return "root"
def get_gain_privilege(isroot, isuser, req_privilege):
    if isroot == "UNCHANGED" and isuser == "UNCHANGED":
        return get_req_privilege(req_privilege)
    elif isroot == True:
        return "root"
    elif isuser == True:
        return "user"
    else:
        return "user"
def retrieve_privileges(vulnID,vulnerabilities):
    for vuln in vulnerabilities:
        if vuln["id"] == vulnID:
            if "cvssMetricV2" in vuln["metrics"]:
                metricV2 = vuln["metrics"]["cvssMetricV2"][0]
                metricCvssV2 = metricV2["cvssData"]
                
                priv_required = get_req_privilege(metricCvssV2["authentication"])
                priv_gained = get_gain_privilege(metricV2["obtainAllPrivilege"],metricV2["obtainUserPrivilege"],metricCvssV2["authentication"])
                return vuln,priv_required,priv_gained
            elif "cvssMetricV30" in vuln["metrics"] or "cvssMetricV31" in vuln["metrics"]: 
                if "cvssMetricV30" in vuln["metrics"]: metricV3 = vuln["metrics"]["cvssMetricV30"][0]
                else: metricV3 = vuln["metrics"]["cvssMetricV31"][0]
                metricCvssV3 = metricV3["cvssData"]

                priv_required = get_req_privilege(metricCvssV3["privilegesRequired"])
                priv_gained = get_gain_privilege(metricCvssV3["scope"],metricCvssV3["scope"],metricCvssV3["privilegesRequired"])
                return vuln,priv_required,priv_gained
            else:
                return vuln,"guest","guest"
            
"""
This function returns the list of vulnerability IDs, given a host
"""
def get_vulns_from_host(host):
    # return host["vulnerabilities"]
    vuln_list = []
    for iface in host["network_interfaces"]:
        for port in iface["ports"]:
            for service in port["services"]:
                vuln_list+=service["cve_list"]
    return list(set(vuln_list))

"""
This function returns the list of cpes, given a host
"""
def get_cpes_from_host(host):
    cpe_list = []
    for iface in host["network_interfaces"]:
        for port in iface["ports"]:
            for service in port["services"]:
                cpe_list+=service["cpe_list"]
    return list(set(cpe_list))

"""
This function returns the credential that is requisite for the given vulnerability
"""
def get_credential_from_vuln(vuln):
    metric = vuln["metrics"]
    if "cvssMetricV2" in metric.keys():
        return metric["cvssMetricV2"][0]["cvssData"]["authentication"]
    elif "cvssMetricV30" in metric.keys():
        return metric["cvssMetricV30"][0]["cvssData"]["attackVector"]
    elif "cvssMetricV31" in metric.keys():
        return metric["cvssMetricV31"][0]["cvssData"]["attackVector"]
    else: return "SINGLE"
    
def build_multiag(network_file,ag_file):
    with open(network_file) as nf:
        content_network = json.load(nf)
    reachability_edges = content_network["edges"]
    devices = content_network["devices"]
    vulnerabilities = content_network["vulnerabilities"]

    G = nx.MultiDiGraph()
    for edge in reachability_edges:
        r_edge = edge["host_link"]
        src_id = r_edge[0]
        dst_id = r_edge[1]
        for host in devices:
            if host["hostname"] == dst_id:
                dst_vulns = get_vulns_from_host(host)
                for v in dst_vulns:
                    vuln,req,gain = retrieve_privileges(v,vulnerabilities)
                    req_state_node = req+"@"+str(src_id)
                    gain_state_node = gain+"@"+str(dst_id)
                    vuln_edge = vuln["id"]
                    if req_state_node not in G.nodes(): G.add_node(req_state_node, type="state")
                    if gain_state_node not in G.nodes(): G.add_node(gain_state_node, type="state")
                    G.add_edge(req_state_node,gain_state_node,vuln=vuln_edge,key=str(src_id)+vuln_edge+str(dst_id))
                    
                    #TODO: add inner edges
                        
    nx.write_graphml_lxml(G, ag_file)
    return G

def build_alertag(folderML):
    G = nx.MultiDiGraph()
    
    for file in os.listdir(folderML):
        # df = pd.read_csv(folderML+"/"+file, encoding='utf-8', errors='ignore')
        with open(folderML+"/"+file, 'r', encoding='utf-8', errors='ignore') as f: df = pd.read_csv(f)
        df = df[df[' Label'] != "BENIGN"]
        df = df[df[' Label'] != "NaN"]
        df = df[df[' Source IP'] != "NaN"]
        df = df[df[' Destination IP'] != "NaN"]
                
        for index, row in df.iterrows():
            srcID = row[" Source IP"]
            dstID = row[" Destination IP"]
            vulnID = str(row[" Protocol"])+"-"+str(row[" Label"])+"-"+str(row[" CWE Flag Count"])
            G.add_edge(srcID,dstID,vuln=vulnID,key=str(srcID)+vulnID+str(dstID))
            
    nx.write_graphml_lxml(G, "data/agAlert.graphml")
    
    G_alert = nx.read_graphml("data/agAlert.graphml")
    if "nan" in G_alert: G_alert.remove_node("nan")
    nx.write_graphml_lxml(G_alert, "data/agAlert.graphml")
    return G_alert

def compute_paths(G,vulnerabilities,filePath,sources=[],goals=[]):
    allGvulns = nx.get_edge_attributes(G, "vuln")
    if len(sources)==0: sources=G.nodes
    else:
        newSources=[]
        for s in sources:
            for n in G.nodes:
                if s in n and n not in newSources: newSources.append(n)
        sources=newSources
    if len(goals)==0: goals=G.nodes
    else:
        newGoals=[]
        for t in goals:
            for n in G.nodes:
                if t in n and n not in newGoals: newGoals.append(n)
        goals=newGoals
        
    attack_paths = []
    startTime = time.perf_counter()
    for s in list(sources):
        for t in list(goals):
            if s==t or s not in G.nodes or t not in G.nodes or not nx.has_path(G,s,t): continue
            # all_paths = list(nx.all_simple_paths(G, source=s, target=t))
            all_paths = list(nx.all_shortest_paths(G, source=s, target=t))
            for p in all_paths:
                vulns_path = []
                for e in G.edges:
                    for i in range(1,len(p)):
                        if e[0] == p[i-1] and e[1]==p[i]:
                            print
                            vulns_path.append(allGvulns[e])
                            # print(allGvulns[e])
                
                # risk_p = {}
                # for e in p: vulns_path.append(allGvulns[e])
                risk_p = compute_risk_analysis(vulns_path,vulnerabilities)
                risk_p["path"] = p
                attack_paths.append(risk_p)
                
        with open(filePath, "w") as outfile:
            json_data = json.dumps({"paths":attack_paths},
                default=lambda o: o.__dict__, indent=2)
            outfile.write(json_data)
    endTime = time.perf_counter()
    
    print(endTime-startTime)

def plot_risk(metric, listNetFile):
    victims_ip={
        "Web srv 16":["192.168.10.50"],
        "Ubu srv 12":["192.168.10.51"],
        "Ubu14 32B":['192.168.10.19'],
        "Ubu14 64B":['192.168.10.17'],
        "Ubu16 32B":['192.168.10.16'],
        "Ubu16 64B":['192.168.10.12'],
        "Win7 Pro 64B":['192.168.10.9'],
        "Win8.1 64B":['192.168.10.5'],
        "WinVista 64B":['192.168.10.8'],
        "Win10pro 32B":['192.168.10.14'],
        "Win10 64B":['192.168.10.15'],
        "MAC":['192.168.10.25']
    }
    
    # victims_risk_alert = {}
    
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
        
        model = pathFile.split("paths/")[1]
        dictVictimRisk[model] = victims_risk
    
    # with open(pathFile) as nf:
    #     paths = json.load(nf)["paths"]
    # for p in paths:
    #     stepsP = p["path"]
    #     for v in victims_ip.keys():
    #         for ipV in victims_ip[v]:
    #             if ipV in stepsP[len(stepsP)-1]:
    #                 if v not in victims_risk.keys(): victims_risk[v] = [p[metric]]
    #                 else: victims_risk[v].append(p[metric])
                    
    # with open(alertPathFile) as nfAl:
    #     pathsAlert = json.load(nfAl)["paths"]
    # for pAl in pathsAlert:
    #     stepsP = pAl["path"]
    #     for v in victims_ip.keys():
    #         for ipV in victims_ip[v]:
    #             if ipV in stepsP[len(stepsP)-1]:
    #                 if v not in victims_risk_alert.keys(): victims_risk_alert[v] = [pAl[metric]]
    #                 else: victims_risk_alert[v].append(pAl[metric])
    
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
    
    # ag_risk_values = []
    # ag_risk_values_alert = []
    # for v in victims:
    #     avgAGrisk = round(stats.mean(victims_risk[v]),2)
    #     ag_risk_values.append(avgAGrisk)
        
    #     avgAGriskAlert = round(stats.mean(victims_risk_alert[v]),2)
    #     ag_risk_values_alert.append(avgAGriskAlert)
            
    # models = {
    #     'AG': ag_risk_values,
    #     'IDS-based AG': ag_risk_values_alert
    # }

    x = np.arange(len(victims))  # the label locations
    num_in_group = len(dictVictimRisk.keys())
    gap = 0.2
    width = (1 - gap) / num_in_group  # the width of the bars
    # width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in models.items():
        # offset = width * multiplier
        offset = width * multiplier - (1 - gap) / 2
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3, fontsize=9, rotation=45)
        multiplier += 1

    ax.set_ylabel('Avg. risk')
    ax.set_xlabel('Victims')
    ax.set_xticks(x + width, victims, rotation=90)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc='upper left')
    ax.set_ylim(0, 2)

    plt.savefig("results/"+metric+".png")

if __name__ == "__main__":
    originalNet = "data/networks/CiC17Net.json"
    onlyAlertNet = "data/networks/alertNet.json"
    partialAlertNet = "data/networks/partialAlertNet.json"
    partialAlertOriginalNet = "data/networks/partialAlertOriginalNet.json"
    fullNet = "data/networks/fullNet.json"
    
    for netfile in [onlyAlertNet,partialAlertNet,partialAlertOriginalNet,fullNet,originalNet]:
        graphfile = netfile.replace("networks","ags").replace(".json","AG.graphml")
        G = build_multiag(netfile,graphfile)
        with open(netfile) as nf:
            vulnerabilities = json.load(nf)["vulnerabilities"]
        pathfile = netfile.replace("networks","paths").replace(".json","Path.json")
        compute_paths(G,vulnerabilities,pathfile,sources=["kali","win81"],
            goals=[
            "192.168.10.50","192.168.10.51",'192.168.10.19','192.168.10.17','192.168.10.16',
            '192.168.10.12','192.168.10.9','192.168.10.5','192.168.10.8','192.168.10.14',
            '192.168.10.15','192.168.10.25'
            ])
    
    # G_std = build_multiag(originalNet)
    # with open(originalNet) as nf:
    #     vulnerabilities = json.load(nf)["vulnerabilities"]
        
    # G_alert = build_multiag(alertNetFile)
    # with open(alertNetFile) as nfAl:
    #     vulnerabilitiesAlert = json.load(nfAl)["vulnerabilities"]
        
    # pathFile = "data/paths.json"
    # alertPathFile = "data/pathsAlert.json"
    # # # compute_paths(G_std,vulnerabilities,pathFile,sources=['guest@fw-205.174.165.80', 'user@fw-205.174.165.80', 'root@fw-205.174.165.80'],
    # # #               goals=['guest@ubu164-192.168.10.16', 'user@win8-192.168.10.5', 'guest@ubu164-192.168.10.12', 'guest@ubu144-192.168.10.19', 'guest@ubu144-192.168.10.17', 'guest@win8-192.168.10.5', 'guest@win7-192.168.10.9', 'user@win7-192.168.10.9', 'user@winvista-192.168.10.8', 'root@winvista-192.168.10.8', 'guest@winvista-192.168.10.8', 'user@win10-192.168.10.14', 'guest@win10-192.168.10.14', 'user@win10-192.168.10.15', 'guest@win10-192.168.10.15', 'guest@mac-192.168.10.25', 'user@ubu164-192.168.10.16', 'user@ubu164-192.168.10.12', 'user@mac-192.168.10.25', 'user@ubu12-192.168.10.51', 'root@ubu12-192.168.10.51', 'guest@ubu12-192.168.10.51', 'user@ubu16-192.168.10.50', 'root@ubu16-192.168.10.50', 'guest@ubu16-192.168.10.50', 'root@mac-192.168.10.25', 'root@win8-192.168.10.5'])
    # # # compute_paths(G_alert,vulnerabilitiesAlert,alertPathFile,sources=['guest@fw-205.174.165.80', 'user@fw-205.174.165.80', 'root@fw-205.174.165.80'],
    # # #               goals=['guest@ubu164-192.168.10.16', 'user@win8-192.168.10.5', 'guest@ubu164-192.168.10.12', 'guest@ubu144-192.168.10.19', 'guest@ubu144-192.168.10.17', 'guest@win8-192.168.10.5', 'guest@win7-192.168.10.9', 'user@win7-192.168.10.9', 'user@winvista-192.168.10.8', 'root@winvista-192.168.10.8', 'guest@winvista-192.168.10.8', 'user@win10-192.168.10.14', 'guest@win10-192.168.10.14', 'user@win10-192.168.10.15', 'guest@win10-192.168.10.15', 'guest@mac-192.168.10.25', 'user@ubu164-192.168.10.16', 'user@ubu164-192.168.10.12', 'user@mac-192.168.10.25', 'user@ubu12-192.168.10.51', 'root@ubu12-192.168.10.51', 'guest@ubu12-192.168.10.51', 'user@ubu16-192.168.10.50', 'root@ubu16-192.168.10.50', 'guest@ubu16-192.168.10.50', 'root@mac-192.168.10.25', 'root@win8-192.168.10.5'])
    # compute_paths(G_std,vulnerabilities,pathFile,sources=["kali","win81"],
    #               goals=[
    #                 "192.168.10.50","192.168.10.51",'192.168.10.19','192.168.10.17','192.168.10.16','192.168.10.12','192.168.10.9','192.168.10.5','192.168.10.8','192.168.10.14','192.168.10.15','192.168.10.25'
    #               ])
    # compute_paths(G_alert,vulnerabilitiesAlert,alertPathFile,sources=["kali","win81"],
    #               goals=[
    #                 "192.168.10.50","192.168.10.51",'192.168.10.19','192.168.10.17','192.168.10.16','192.168.10.12','192.168.10.9','192.168.10.5','192.168.10.8','192.168.10.14','192.168.10.15','192.168.10.25'
    #               ])
    
    plot_risk("risk",[onlyAlertNet,partialAlertNet,partialAlertOriginalNet,fullNet,originalNet])
    # plot_risk("risk",pathFile,alertPathFile) #{impact, likelihood, risk}
    # plot_risk("impact",pathFile,alertPathFile)
    # plot_risk("likelihood",pathFile,alertPathFile)
    
