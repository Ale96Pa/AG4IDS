import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt
from ag_utils import merge_all_edges, find_node_from_ip, merge_all_nodes_by_priviledge



G = nx.read_graphml('data/ags/alertNetAG.graphml') # alertNetAG.graphml or CiC17NetAG.graphml
print('Nodes: {}'.format(list(G.nodes(data=True))))
print('Edges: {}'.format(list(G.edges(data=True))))

print('Number of nodes: {}'.format(len(list(G.nodes()))))
print('Number of edges: {}'.format(len(list(G.edges()))))
print('Number of nodes: {}'.format(G.number_of_nodes()))
print('Number of edges: {}'.format(G.number_of_edges()))
G = merge_all_edges(G)
print('Number of nodes: {}'.format(G.number_of_nodes()))
print('Number of edges: {}'.format(G.number_of_edges()))



# for source in G.nodes():
#     for target in G.nodes():
#         if not source == target:
#             print('Has path from "{}" to "{}" gives: {}'.format(source, target, nx.has_path(G, source, target)))


# paths = []
# total_paths = 0
# existing_paths = 0
# for n1 in G.nodes():
#     for n2 in G.nodes():
#         if not n1 == n2:
#             total_paths += 1
#             try:
#                 shortest_path = nx.shortest_path(G, source=n1, target=n2)
#                 shortest_path_vulns = []
#                 for node_id in range(len(shortest_path) - 1):
#                     step_source = shortest_path[node_id]
#                     step_destination = shortest_path[node_id+1]
#                     shortest_path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
#                 shortest_path_length = nx.shortest_path_length(G, source=n1, target=n2)
#                 all_paths = list(nx.all_simple_paths(G, source=n1, target=n2))
#                 all_paths_vulns = []
#                 for path in all_paths:
#                     path_vulns = []
#                     for node_id in range(len(path) - 1):
#                         step_source = path[node_id]
#                         step_destination = path[node_id+1]
#                         path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
#                     all_paths_vulns.append(path_vulns)
#                 existing_paths += 1
#                 paths.append({'source': n1, 'target': n2, 'shortest_path': shortest_path, 'length': shortest_path_length, 'shortest_path_vulns': shortest_path_vulns, 'all_paths': all_paths, 'all_paths_vulns': all_paths_vulns})
#             except nx.NetworkXNoPath:
#                 continue
# print('paths: {}'.format(paths[-1]))
# print('total_paths: {}'.format(total_paths))
# print('existing_paths: {}'.format(existing_paths))

# all_paths_dict = dict(nx.all_pairs_shortest_path(G))
# # print('{}'.format(all_paths_dict['guest@ubu144-192.168.10.19']['user@win7-192.168.10.9']))


# sources = ['root@fw-205.174.165.80', 'user@fw-205.174.165.80', 'guest@fw-205.174.165.80']
# targets = ['root@ubu16-192.168.10.50', 'user@ubu16-192.168.10.50', 'guest@ubu16-192.168.10.50']
# for source in sources:
#     for target in targets:
#         if not source == target and 'kali' not in target and 'win81' not in target:
#             print('Has path from "{}" to "{}" gives: {}'.format(source, target, nx.has_path(G, source, target)))


# source = 'guest@fw-205.174.165.80'
# target = 'user@ubu16-192.168.10.50'
# shortest_path = nx.shortest_path(G, source=source, target=target)
# shortest_path_vulns = []
# for node_id in range(len(shortest_path) - 1):
#     step_source = shortest_path[node_id]
#     step_destination = shortest_path[node_id+1]
#     shortest_path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
# print('From "{}" to "{}" the shortest path is {}, which requires exploiting the following sequence of vulnerabilities: {}'.format(source, target, shortest_path, shortest_path_vulns))


G_new = merge_all_nodes_by_priviledge(G)

print('Number of nodes: {}'.format(G_new.number_of_nodes()))
print('Number of edges: {}'.format(G_new.number_of_edges()))
print('Edges: {}'.format(list(G_new.edges(data=True))))
print('Edges: {}'.format(list(G_new.edges(data=False))))
print(len(list(G_new.edges(data=False))))
print(len(list(set(list(G_new.edges(data=False))))))


node = find_node_from_ip(G_new, ip='205.174.165.80')
print(node)


# if G.has_edge('guest@win81-205.174.165.69', 'guest@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['guest@win81-205.174.165.69', 'guest@fw-205.174.165.80']))
# if G.has_edge('guest@win81-205.174.165.69', 'root@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['guest@win81-205.174.165.69', 'root@fw-205.174.165.80']))
# if G.has_edge('guest@win81-205.174.165.69', 'user@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['guest@win81-205.174.165.69', 'user@fw-205.174.165.80']))
# if G.has_edge('root@win81-205.174.165.69', 'guest@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['root@win81-205.174.165.69', 'guest@fw-205.174.165.80']))
# if G.has_edge('root@win81-205.174.165.69', 'root@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['root@win81-205.174.165.69', 'root@fw-205.174.165.80']))
# if G.has_edge('root@win81-205.174.165.69', 'user@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['root@win81-205.174.165.69', 'user@fw-205.174.165.80']))
# if G.has_edge('user@win81-205.174.165.69', 'guest@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['user@win81-205.174.165.69', 'guest@fw-205.174.165.80']))
# if G.has_edge('user@win81-205.174.165.69', 'root@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['user@win81-205.174.165.69', 'root@fw-205.174.165.80']))
# if G.has_edge('user@win81-205.174.165.69', 'user@fw-205.174.165.80'):
#     print('privileges on: {}'.format(G.edges['user@win81-205.174.165.69', 'user@fw-205.174.165.80']))
# print('privileges off: {}'.format(G_new.edges['win81-205.174.165.69', 'fw-205.174.165.80']))


# Draw the AG
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 10]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 10]
pos = nx.random_layout(G)
nx.draw_networkx_nodes(G, pos=pos)
nx.draw_networkx_labels(G, pos=pos)
# nx.draw_networkx_edges(G, pos=pos)
nx.draw_networkx_edges(G, pos, edgelist=elarge, edge_color="b", width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)
# nx.draw_networkx_edge_labels(G, pos=pos)
plt.savefig('ag.png')
plt.show()

# Draw the AG
# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 10]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 10]
pos = nx.random_layout(G_new)
nx.draw_networkx_nodes(G_new, pos=pos)
nx.draw_networkx_labels(G_new, pos=pos)
nx.draw_networkx_edges(G_new, pos=pos)
# nx.draw_networkx_edges(G_new, pos, edgelist=elarge, edge_color="b", width=6)
# nx.draw_networkx_edges(
#     G_new, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
# )
# nx.draw_networkx_edge_labels(G, pos=pos)
plt.savefig('ag.png')
plt.show()

# print('\nnodes: {}'.format(G_new.nodes()))
# print('\n\n')
# source = 'fw-205.174.165.80'
# targets = ['ubu16-192.168.10.50', 'ubu12-192.168.10.51', 'winvista-192.168.10.8', 'win10-192.168.10.15', 'win7-192.168.10.9', 'win10-192.168.10.14']
# for target in targets:
#     if nx.has_path(G_new, source=source, target=target):
#         shortest_path = nx.shortest_path(G_new, source=source, target=target)
#         shortest_path_vulns = []
#         for node_id in range(len(shortest_path) - 1):
#             step_source = shortest_path[node_id]
#             step_destination = shortest_path[node_id+1]
#             shortest_path_vulns.append(G_new.get_edge_data(step_source, step_destination)['vulns'])
#         print('From "{}" to "{}" the shortest path is {}, which requires exploiting the following sequence of vulnerabilities: {}'.format(source, target, shortest_path, shortest_path_vulns))
#     else:
#         print('No attack path found from "{}" to "{}"'.format(source, target))

# print('\n\n')
# sources = ['root@fw-205.174.165.80', 'user@fw-205.174.165.80', 'guest@fw-205.174.165.80']
# targets = ['root@winvista-192.168.10.8', 'user@winvista-192.168.10.8', 'guest@winvista-192.168.10.8']
# for source in sources:
#     for target in targets:
#         if nx.has_path(G, source=source, target=target):
#             shortest_path = nx.shortest_path(G, source=source, target=target)
#             shortest_path_vulns = []
#             for node_id in range(len(shortest_path) - 1):
#                 step_source = shortest_path[node_id]
#                 step_destination = shortest_path[node_id+1]
#                 shortest_path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
#             print('From "{}" to "{}" the shortest path is {}, which requires exploiting the following sequence of vulnerabilities: {}'.format(source, target, shortest_path, shortest_path_vulns))
#         else:
#             print('No attack path found from "{}" to "{}"'.format(source, target))

# print('\n\n')
# sources = ['root@fw-205.174.165.80', 'user@fw-205.174.165.80', 'guest@fw-205.174.165.80']
# for source in sources:
#     outing_edges = G.edges(source)
#     destinations = [edge[1] for edge in outing_edges]
#     # print('Edges coming out of "{}" are: {}'.format(source, G.edges(source)))
#     if len(destinations) == 0:
#         print('No edges coming out of "{}"'.format(source))
#     else:
#         print('Edges coming out of "{}" link to: {}'.format(source, destinations))


# print('\n\n')
# sources = ['fw-205.174.165.80']
# for source in sources:
#     outing_edges = G_new.edges(source)
#     destinations = [edge[1] for edge in outing_edges]
#     # print('Edges coming out of "{}" are: {}'.format(source, G.edges(source)))
#     if len(destinations) == 0:
#         print('No edges coming out of "{}"'.format(source))
#     else:
#         print('Edges coming out of "{}" link to: {}'.format(source, destinations))

# print('\n\n')
# destinations = ['root@fw-205.174.165.80', 'user@fw-205.174.165.80', 'guest@fw-205.174.165.80']
# for destination in destinations:
#     incoming_edges = G.in_edges(destination)
#     sources = [edge[0] for edge in incoming_edges]
#     # print('Edges coming out of "{}" are: {}'.format(source, G.edges(source)))
#     if len(sources) == 0:
#         print('No edges incoming in "{}"'.format(destination))
#     else:
#         print('Edges incoming in "{}" are from: {}'.format(destination, sources))

# print('\n\n')
# destinations = ['fw-205.174.165.80']
# for destination in destinations:
#     incoming_edges = G_new.in_edges(destination)
#     sources = [edge[0] for edge in incoming_edges]
#     # print('Edges coming out of "{}" are: {}'.format(source, G.edges(source)))
#     if len(sources) == 0:
#         print('No edges incoming in "{}"'.format(destination))
#     else:
#         print('Edges incoming in "{}" are from: {}'.format(destination, sources))

# print('\n\n')
# # G = nx.dodecahedral_graph()
# # nx.draw(G)
# # plt.show()

# # A = nx.nx_agraph.to_agraph(G)

# # G = pgv.AGraph('data/ag.graphml')
# # A.draw()