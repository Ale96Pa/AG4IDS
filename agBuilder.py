import json, math, time, csv, sys, os
import pandas as pd
import networkx as nx
import numpy as np
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
    
    lik_risk = 0
    MTAO = sum(lambda_exploit_scores)
    # MTAO = (sum(lambda_exploit_scores)-min(lambda_exploit_scores))/sum(lambda_exploit_scores)
    if MTAO>0:
        argument = (MTAO-min(lambda_exploit_scores))/MTAO
        lik_risk = (-20)*math.log(argument,10)
            
    # lik_risk = (sum(lambda_exploit_scores)-min(lambda_exploit_scores))/sum(lambda_exploit_scores) if len(lambda_exploit_scores)>0 else 0
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
    
def build_multiag(network_file):
    with open(network_file) as nf:
        content_network = json.load(nf)
    reachability_edges = content_network["edges"]
    devices = content_network["devices"]
    vulnerabilities = content_network["vulnerabilities"]

    G = nx.MultiDiGraph()
    for r_edge in reachability_edges:
        src_id = r_edge[0]
        dst_id = r_edge[1]
        for host in devices:
            if host["id"] == dst_id:
                dst_vulns = get_vulns_from_host(host)
                for v in dst_vulns:
                    vuln,req,gain = retrieve_privileges(v,vulnerabilities)
                    req_state_node = req+"@"+str(src_id)
                    gain_state_node = gain+"@"+str(dst_id)
                    vuln_edge = vuln["id"]
                    if req_state_node not in G.nodes(): G.add_node(req_state_node, type="state")
                    if gain_state_node not in G.nodes(): G.add_node(gain_state_node, type="state")
                    G.add_edge(req_state_node,gain_state_node,vuln=vuln_edge,key=str(src_id)+vuln_edge+str(dst_id))

                        
    nx.write_graphml_lxml(G, "data/ag.graphml")
    return G

if __name__ == "__main__":
    G = build_multiag("data/CiC17Net.json")