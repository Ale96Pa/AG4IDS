import json, os, requests, time
import networkx as nx
import pandas as pd

URL_CVE = "https://services.nvd.nist.gov/rest/json/cves/2.0?"
SERVICES={
    "kali":"kali 2025.1a",
    "win81":"windows 8.1",
    "win8":"windows 8",
    "dns":"Win Server 2016",
    "ubu12":"Ubuntu 12",
    "ubu16":"Ubuntu 16",
    "ubu164":"Ubuntu 16.4",
    "ubu144":"Ubuntu 14.4",
    "win7":"Windows 7 pro",
    "winvista":"Windows vista",
    "win10":"Windows 10 pro",
    "fw": "fortinet",
    "mac": "macOS Ventura 13.6.1"
}

def get_dump_nvd(vulnFile):
    dump_cve={}
    headers = {'content-type': 'application/json'}
    for service in SERVICES.keys():
        dump_cve[service] = []
        params={
            "keywordSearch": SERVICES[service],
        }
        time.sleep(6)
        response = requests.get(URL_CVE, params=params,  headers=headers)
        if response.status_code == 200:
            jsonResponse = response.json()
            for cve in jsonResponse["vulnerabilities"]:
                dump_cve[service].append(cve["cve"])
    
    with open(vulnFile, "w") as outfile:
        json_data = json.dumps({"services":dump_cve}, 
            default=lambda o: o.__dict__, indent=2)
        outfile.write(json_data)
    
    # return dump_cve

def get_dump_cveList(ruleFolder, vulnFile):
    dictAttackCve = {}
    for ruleFile in os.listdir(ruleFolder):
        with open(ruleFolder+ruleFile, 'r', encoding='utf-8', errors='ignore') as f:
            if "ftp" in ruleFile: attackType = "ftp"
            elif "dos" in ruleFile: attackType = "dos"
            dictAttackCve[attackType] = []
            
            for line in f:
                if "cve" in line:
                    for field in line.split(";"):
                        if "cve" in field:
                            if "reference" in field:
                                cveID = field.split(":")[1].replace("cve,","CVE-")
                                if "CVE-CVE-" in cveID: cveID.replace("CVE-CVE-","CVE-")
                                if "CVE-" in cveID and cveID not in dictAttackCve[attackType]:
                                    dictAttackCve[attackType].append(cveID)
                            elif "metadata" in field:
                                for subfield in field.split(","):
                                    if "cve" in subfield: 
                                        cveID = subfield.split("cve ")[1].replace("_","-")
                                        if "CVE-CVE-" in cveID: cveID.replace("CVE-CVE-","CVE-")
                                        if "CVE-" in cveID and cveID not in dictAttackCve[attackType]: 
                                            dictAttackCve[attackType].append(cveID)
    
    dump_cve={}
    headers = {'content-type': 'application/json'}
    for attack in dictAttackCve.keys():
        dump_cve[attack] = []
        for cve in dictAttackCve[attack]:
            params={
                "cveId": cve,
            }
            time.sleep(6)
            response = requests.get(URL_CVE, params=params,  headers=headers)
            if response.status_code == 200:
                jsonResponse = response.json()
                for cve in jsonResponse["vulnerabilities"]:
                    dump_cve[attack].append(cve["cve"])
    
    with open(vulnFile, "w") as outfile:
        json_data = json.dumps({"attacks":dump_cve}, 
            default=lambda o: o.__dict__, indent=2)
        outfile.write(json_data)


def generate_devices(network_file,vulnerabilityFile):
    all_devs=[]
    vulnerabilities=[]
    
    for service in SERVICES:
        vulns_srv, vulnsObjs = getVulnsByService(service,vulnerabilityFile)
        vulnerabilities+=vulnsObjs
        ips=[]
        if service=="kali":
            ips = ["205.174.165.73"]
            hostname = "entryPoint"
            typeHost = "laptop"
        elif service=="win81":
            ips = ["205.174.165.69","205.174.165.70","205.174.165.71"]
            hostname = "entryPoint"
            typeHost = "laptop"
        elif service=="dns":
            ips = ["192.168.10.3"]
            hostname = "DNSServer"
            typeHost = "server"
        elif service=="ubu12":
            ips = ["192.168.10.51"]#,"205.174.165.66"]
            hostname = "UbuntuServer"
            typeHost = "server"
        elif service=="ubu16":
            ips = ["192.168.10.50"]#,"205.174.165.68"]
            hostname = "WebServer"
            typeHost = "server"
        elif service=="win8":
            ips = ["192.168.10.5"]
            hostname = "workstationWin8"
            typeHost = "client"
        elif service=="ubu164":
            ips = ["192.168.10.16","192.168.10.12"]
            hostname = "workstationUbuntu16"
            typeHost = "client"
        elif service=="ubu144":
            ips = ["192.168.10.19","192.168.10.17"]
            hostname = "workstationUbuntu14"
            typeHost = "client"
        elif service=="win7":
            ips = ["192.168.10.9"]
            hostname = "workstationWindows7"
            typeHost = "client"
        elif service=="winvista":
            ips = ["192.168.10.8"]
            hostname = "workstationWindowsVista"
            typeHost = "client"
        elif service=="win10":
            ips = ["192.168.10.14","192.168.10.15"]
            hostname = "workstationWindows10"
            typeHost = "client"
        elif service=="fw":
            ips = ["205.174.165.80"]
            hostname = "Fortinet"
            typeHost = "firewall"
        elif service=="mac":
            ips = ["192.168.10.25"]
            hostname = "MAC"
            typeHost = "client"
        
        for ip in ips:
            all_devs.append(
                {
                    "id": hostname+"-"+service+"-"+ip,
                    "hostname": service+"-"+ip,
                    "type": typeHost,
                    "network_interfaces": [
                        {
                            "ipaddress": ip,
                            "macaddress": "ff:ff:ff:ff:ff:ff",
                            "ports": [
                                {
                                    "number": 0,
                                    "state": "open",
                                    "protocol": "TCP",
                                    "services": [
                                        {
                                            "name": "all",
                                            "cpe_list": [],
                                            "cve_list": vulns_srv
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "local_applications": []
                }     
        )
    
    edges=[]
    for devSrc in all_devs:
        idSrc = devSrc["id"]
        typeSrc = devSrc["type"]
        hostSrc = devSrc["hostname"]
        for devDst in all_devs:
            idDst = devDst["id"]
            typeDst = devDst["type"]
            hostDst = devDst["hostname"]
            if idSrc==idDst: continue
            
            if "entryPoint" in idSrc and "firewall" in typeDst:
                edges.append({"host_link":[hostSrc,hostDst]})
                edges.append({"host_link":[hostDst,hostSrc]})
            # if "kali" in idSrc and "workstationWindows" in idDst:
            #     edges.append({"host_link":[hostSrc,hostDst])
            #     edges.append({"host_link":[hostDst,hostSrc])
            if "client" in typeSrc and "client" in typeDst:
                edges.append({"host_link":[hostSrc,hostDst]})
                edges.append({"host_link":[hostDst,hostSrc]})
            
    edges.append({"host_link":["fw-205.174.165.80","ubu12-192.168.10.51"]})
    edges.append({"host_link":["ubu12-192.168.10.51","fw-205.174.165.80"]})
    edges.append({"host_link":["fw-205.174.165.80","ubu16-192.168.10.50"]})
    edges.append({"host_link":["ubu16-192.168.10.50","fw-205.174.165.80"]})
    
    edges.append({"host_link":["dns-192.168.10.3","ubu12-192.168.10.51"]})
    edges.append({"host_link":["ubu12-192.168.10.51","dns-192.168.10.3"]})
    edges.append({"host_link":["dns-192.168.10.3","ubu16-192.168.10.50"]})
    edges.append({"host_link":["ubu16-192.168.10.50","dns-192.168.10.3"]})
    
    edges.append({"host_link":["fw-205.174.165.80","mac-192.168.10.25"]})
    edges.append({"host_link":["mac-192.168.10.25","fw-205.174.165.80"]})
    edges.append({"host_link":["fw-205.174.165.80","win8-192.168.10.5"]})
    edges.append({"host_link":["win8-192.168.10.5","fw-205.174.165.80"]})
    
    
    edgesG=[]
    for e in edges:
        edgesG.append(e["host_link"])
    G = nx.DiGraph()
    G.add_edges_from(edgesG)
    nx.write_graphml(G, "data/networkICIDS.graphml")
            
    with open(network_file, "w") as outfile:
        json_data = json.dumps({"devices":all_devs,
                                "vulnerabilities":vulnerabilities,
                                "edges":edges}, 
            default=lambda o: o.__dict__, indent=2)
        outfile.write(json_data)
    
import random
def getVulnsByService(servicename, vulnfile):
    with open(vulnfile) as vulnf:
        services = json.load(vulnf)["services"]
    
    if servicename == "win8": servicename="win81"
    vulnsID=[]
    for cve in services[servicename]: vulnsID.append(cve["id"])
    # return vulnsID, services[servicename]
    if len(vulnsID)>7: return random.sample(vulnsID, 7), services[servicename]
    else: return vulnsID, services[servicename]

def getVulnsByAlert(alertFolder, vulnFile, netFile, newNetFile):
    dictDevAttack = {}
    allAttacks = []
    
    for alertFile in os.listdir(alertFolder):
        print(alertFile)
        
        with open(alertFolder+alertFile, 'r', encoding='utf-8', errors='ignore') as f: df = pd.read_csv(f, low_memory=False)
        for index, row in df.iterrows():
            ipAddr = row[' Destination IP']
            attackType = row[' Label']
            if type(attackType) != str: continue
            
            if "DoS" in attackType: attackType="dos"
            if "DDoS" in attackType: attackType="dos"
            if "FTP" in attackType: attackType="ftp"
            if ipAddr=="172.16.0.1": ipAddr="205.174.165.80"
            
            if attackType == "BENIGN": continue
            if attackType == "NaN": continue
            if ipAddr == "NaN": continue
            if attackType not in ["DoS","FTP","dos","ftp"]: continue
            
            if ipAddr not in dictDevAttack.keys(): dictDevAttack[ipAddr] = [attackType]
            else: 
                if attackType not in dictDevAttack[ipAddr]: dictDevAttack[ipAddr].append(attackType)
            
            if attackType not in allAttacks: allAttacks.append(attackType)
    
    print(set(allAttacks))
    print(dictDevAttack)
    
    with open(vulnFile) as vulnf: attacks = json.load(vulnf)["attacks"]
    with open(netFile) as netf: content = json.load(netf)
    
    devices = content["devices"]
    vulnerabilities = content["vulnerabilities"]
    edges = content["edges"]
    
    all_devs=[]
    added_vulns=[]
    for dev in devices:
        ifaceObjs=[]
        for iface in dev["network_interfaces"]:
            ip = iface["ipaddress"]
            cveAttackList=[]
            if ip in dictDevAttack.keys():
                for atck in dictDevAttack[ip]:
                    for cve in attacks[atck]:
                        if cve["id"] not in cveAttackList:
                            cveAttackList.append(cve["id"])
                            added_vulns.append(cve)
            for port in iface["ports"]:
                for srv in port["services"]:
                    oldCve = srv["cve_list"]
                    allCVE = oldCve+cveAttackList
            
            ifaceObjs.append({
                "ipaddress": ip,
                "macaddress": "ff:ff:ff:ff:ff:ff",
                "ports": [
                    {
                        "number": 0,
                        "state": "open",
                        "protocol": "TCP",
                        "services": [
                            {
                                "name": "all",
                                "cpe_list": [],
                                "cve_list": allCVE
                            }
                        ]
                    }
                ]
                }
            )
        
        all_devs.append(
                {
                    "id": dev["id"],
                    "hostname": dev["hostname"],
                    "type": dev["type"],
                    "network_interfaces": ifaceObjs,
                    "local_applications": []
                }     
        )
    
    with open(newNetFile, "w") as outfile:
        json_data = json.dumps({"devices":all_devs,
                                "vulnerabilities":vulnerabilities+added_vulns,
                                "edges":edges}, 
            default=lambda o: o.__dict__, indent=2)
        outfile.write(json_data)
    

if __name__=="__main__":
    network_file = "data/CiC17Net.json"
    alertNetworkFile = "data/CiC17NetAlert.json"
    
    """First generation of network inventory from CIC-IDS"""
    # vulnerabilityFile = "data/vulns.json"
    # get_dump_nvd(vulnerabilityFile)
    # listCve = getVulnsByService("ubu16")
    # generate_devices(network_file,vulnerabilityFile)
    
    """Retrieving vulnerabilities from the rule dataset"""
    # rule_folder = "emerging_rules/"
    # get_dump_cveList(rule_folder, "vulnsAttack.json")
    
    """Build the alert-based network inventory"""
    vulnAttackFile = "data/vulnsAttack.json"
    alert_folder = "data/TrafficLabelling/"
    getVulnsByAlert(alert_folder, vulnAttackFile, network_file, alertNetworkFile)
