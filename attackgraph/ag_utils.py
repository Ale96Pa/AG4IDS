import os, sys
import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from ids.ids_utils import get_all_flows_from_data


def merge_nodes(G, nodes, new_node):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """

    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))
    G_new.add_edges_from(list(G.edges(data=True)))
    
    G_new.add_node(new_node) # Add the 'merged' node
    
    for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            if not G_new.has_edge(new_node, n2):
                G_new.add_edge(new_node, n2, vulns = data['vulns'], weight = data['weight'])
            else:
                prev_vulns = G_new.edges[new_node, n2]['vulns']
                new_vulns = list(set(prev_vulns + data['vulns']))
                new_weight = len(new_vulns)
                G_new.remove_edge(new_node, n2)
                G_new.add_edge(new_node, n2, vulns = new_vulns, weight = new_weight)
        elif n2 in nodes:
            if not G_new.has_edge(n1, new_node):
                G_new.add_edge(n1, new_node, vulns = data['vulns'], weight = data['weight'])
            else:
                prev_vulns = G_new.edges[n1, new_node]['vulns']
                new_vulns = list(set(prev_vulns + data['vulns']))
                new_weight = len(new_vulns)
                G_new.remove_edge(n1, new_node)
                G_new.add_edge(n1, new_node, vulns = new_vulns, weight = new_weight)
    
    for n in nodes: # remove the merged nodes
        G_new.remove_node(n)

    return G_new


def merge_all_nodes_by_priviledge(G):
    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))
    G_new.add_edges_from(list(G.edges(data=True)))

    nodes = list(G_new.nodes(data=False))
    unique_nodes = list(set([node.split('@')[-1] for node in nodes]))
    mapping_unique_nodes = {unique_node: [node for node in nodes if unique_node in node] for unique_node in unique_nodes}
    for device_node, privilege_nodes in mapping_unique_nodes.items():
        G_new = merge_nodes(G_new, privilege_nodes, device_node)
    return G_new


def merge_all_edges(G):
    edges_with_data = G.edges(data=True)
    edges_without_data = G.edges(data=False)
    unique_edges = list(set(edges_without_data))
    merged_edges_with_data = []
    for edge in unique_edges:
        vulns = []
        for e in edges_with_data:
            if e[0] == edge[0] and e[1] == edge[1]:
                vulns.append(e[2]['vuln'])
        merged_edges_with_data.append((edge[0], edge[1], {'vulns': vulns, 'weight': len(vulns)}))
    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))
    G_new.add_edges_from(merged_edges_with_data)
    return G_new


def find_node_from_ip(G, ip, verbose=False):
    nodes = list(G.nodes())
    if verbose:
        print('Looking for IP "{}" in list of nodes {}...'.format(ip, nodes))
    if ip == '172.16.0.1':
        ip = '205.174.165.80'
    found_nodes = [node for node in nodes if ip==node]
    if len(found_nodes) == 0:
        if verbose:
            print('Did not find any node with IP "{}"'.format(ip))
        return None
    if len(found_nodes) > 1:
        raise ValueError('Found more than one node with same IP. Something is wrong!')
    else:
        return found_nodes[0]


def keep_only_ips_in_nodes_names(G):
    # print('Number of nodes before renaming: {}'.format(G.number_of_nodes()))
    # print('Number of edges before renaming: {}'.format(G.number_of_edges()))
    # print('Nodes: {}'.format(list(G.nodes(data=True))))
    # print('Edges: {}'.format(list(G.edges(data=True))))
    G = nx.relabel_nodes(G, lambda x: x.split('-')[-1])
    # print('Number of nodes after renaming: {}'.format(G.number_of_nodes()))
    # print('Number of edges after renaming: {}'.format(G.number_of_edges()))
    # print('Nodes: {}'.format(list(G.nodes(data=True))))
    # print('Edges: {}'.format(list(G.edges(data=True))))
    return G


def augment_ag(G, path_prob, flows):
    # print('Number of nodes before random addition: {}'.format(G.number_of_nodes()))
    # print('Number of edges before random addition: {}'.format(G.number_of_edges()))
    for flow in flows:
        if np.random.uniform(0.0, 1.0) < path_prob:
            source = flow.split('-')[0]
            destination = flow.split('-')[1]
            if find_node_from_ip(G, source) is None:
                G.add_node(source)
            else:
                source = find_node_from_ip(G, source)
            if find_node_from_ip(G, destination) is None:
                G.add_node(destination)
            else:
                destination = find_node_from_ip(G, destination)
            if not G.has_edge(source, destination):
                G.add_edge(source, destination)
    # print('Number of nodes after random addition: {}'.format(G.number_of_nodes()))
    # print('Number of edges after random addition: {}'.format(G.number_of_edges()))
    return G



def import_ag(args):
    file = '{}.graphml'.format(os.path.join(args.ags_folder, args.ag))
    G = nx.read_graphml(file)
    G = merge_all_edges(G)
    G = merge_all_nodes_by_priviledge(G)
    G = keep_only_ips_in_nodes_names(G)
    if args.ag_path_prob > 0:
        flows = get_all_flows_from_data(args)
        G = augment_ag(G, args.ag_path_prob, flows)
    return G
