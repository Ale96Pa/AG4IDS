import os, sys, json
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from buildIDSnet import internal_net,get_dump_nvd,getVulnsByService,generate_devices, get_dump_cveList, getVulnsByAlert
from attackgraph.agBuilder import build_multiag,compute_paths,plot_risk
import ids.run_all as ra

originalNet = "data/networks/CiC17Net.json"
onlyAlertNet = "data/networks/alertNet.json"
partialAlertNet = "data/networks/partialAlertNet.json"
partialAlertOriginalNet = "data/networks/partialAlertOriginalNet.json"
fullNet = "data/networks/fullNet.json"

if __name__=="__main__":
    
    """
    PREPARATION: dataset preprocessing
    """
    
    internal_net(originalNet)
        
    """First generation of network inventory from CIC-IDS"""
    vulnerabilityFile = "data/vulns.json"
    get_dump_nvd(vulnerabilityFile)
    listCve = getVulnsByService("ubu16")
    generate_devices(originalNet,vulnerabilityFile)
    
    """Retrieving vulnerabilities from the rule dataset"""
    rule_folder = "emerging_rules/"
    get_dump_cveList(rule_folder, "data/vulnsAttack.json")
    
    """Build the alert-based network inventory"""
    vulnAttackFile = "data/vulnsAttack.json"
    alert_folder = "data/TrafficLabelling/"
    getVulnsByAlert(alert_folder, vulnAttackFile, originalNet, 
            [onlyAlertNet,partialAlertNet,partialAlertOriginalNet,fullNet])
    
    """
    ATTACK GRAPH GENERATION
    """
    for netfile in [originalNet,partialAlertOriginalNet,fullNet,onlyAlertNet,partialAlertNet]:
        graphfile = netfile.replace("networks","ags").replace(".json","AG.graphml")
        G = build_multiag(netfile,graphfile)
        with open(netfile) as nf:
            vulnerabilities = json.load(nf)["vulnerabilities"]
        pathfile = netfile.replace("networks","paths").replace(".json","Path.json")
        compute_paths(G,vulnerabilities,pathfile,sources=[
            "kali","fw","win81"
            ],
            goals=[
            "192.168.10.50","192.168.10.51",'192.168.10.19','192.168.10.17','192.168.10.16',
            '192.168.10.12','192.168.10.9','192.168.10.5','192.168.10.8','192.168.10.14',
            '192.168.10.15','192.168.10.25'
            ])
    
    plot_risk("likelihood",[originalNet,partialAlertOriginalNet,fullNet])
    
    """
    AG-integrated IDS
    """
    ra.run_all_blue_box()
    ra.run_all_blue_box_parallel()
    ra.run_all_blue_box_controlled()
    
    """
    AG-based IDS refinement
    """
    ra.run_all_orange_box()
    ra.run_all_orange_box_parallel()
    ra.run_all_orange_box_controlled()
    