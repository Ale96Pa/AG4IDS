import json, os, requests, time
import networkx as nx

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
                    "id": service+"-"+ip,
                    "hostname": hostname,
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
            
            if "entryPoint" in hostSrc and "firewall" in typeDst:
                edges.append([idSrc,idDst])
                edges.append([idDst,idSrc])
            if "kali" in idSrc and "workstationWindows" in devDst["hostname"]:
                edges.append([idSrc,idDst])
                edges.append([idDst,idSrc])
            if "client" in typeSrc and "client" in typeDst:
                edges.append([idSrc,idDst])
                edges.append([idDst,idSrc])
            
    edges.append(["fw-205.174.165.80","ubu12-192.168.10.51"])
    edges.append(["ubu12-192.168.10.51","fw-205.174.165.80"])
    edges.append(["fw-205.174.165.80","ubu16-192.168.10.50"])
    edges.append(["ubu16-192.168.10.50","fw-205.174.165.80"])
    
    edges.append(["dns-192.168.10.3","ubu12-192.168.10.51"])
    edges.append(["ubu12-192.168.10.51","dns-192.168.10.3"])
    edges.append(["dns-192.168.10.3","ubu16-192.168.10.50"])
    edges.append(["ubu16-192.168.10.50","dns-192.168.10.3"])
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nx.write_graphml(G, "data/networkICIDS.graphml")
            
    with open(network_file, "w") as outfile:
        json_data = json.dumps({"devices":all_devs,
                                "vulnerabilities":vulnerabilities,
                                "edges":edges}, 
            default=lambda o: o.__dict__, indent=2)
        outfile.write(json_data)
    

def getVulnsByService(servicename, vulnfile):
    with open(vulnfile) as vulnf:
        services = json.load(vulnf)["services"]
    
    if servicename == "win8": servicename="win81"
    vulnsID=[]
    for cve in services[servicename]: vulnsID.append(cve["id"])
    return vulnsID, services[servicename]


if __name__=="__main__":
    vulnerabilityFile = "data/vulns.json"
    network_file = "data/CiC17Net.json"
    # get_dump_nvd(vulnerabilityFile)
    # listCve = getVulnsByService("ubu16")
    generate_devices(network_file,vulnerabilityFile)
