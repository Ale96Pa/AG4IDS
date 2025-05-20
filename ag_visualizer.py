import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt



G = nx.read_graphml('data/ag.graphml')
print('Nodes: {}'.format(list(G.nodes(data=True))))
print('Edges: {}'.format(list(G.edges(data=True))))

print('Number of nodes: {}'.format(len(list(G.nodes()))))
print('Number of edges: {}'.format(len(list(G.edges()))))
print('Number of nodes: {}'.format(G.number_of_nodes()))
print('Number of edges: {}'.format(G.number_of_edges()))
edges_with_data = G.edges(data=True)
edges_without_data = G.edges(data=False)
unique_edges = list(set(edges_without_data))
# print(edges_with_data)
# print(unique_edges)
print(len(list(set(edges_without_data))))
merged_edges_with_data = []
for edge in unique_edges:
    vulns = []
    for e in edges_with_data:
        if e[0] == edge[0] and e[1] == edge[1]:
            vulns.append(e[2]['vuln'])
    merged_edges_with_data.append((edge[0], edge[1], {'vulns': vulns, 'weight': len(vulns)}))
# print(merged_edges_with_data)
print(len(merged_edges_with_data))


G_new = nx.DiGraph()
G_new.add_nodes_from(list(G.nodes(data=True)))
G_new.add_edges_from(merged_edges_with_data)
G = G_new
print('Number of nodes: {}'.format(G.number_of_nodes()))
print('Number of edges: {}'.format(G.number_of_edges()))


for source in G.nodes():
    for target in G.nodes():
        if not source == target:
            print('Has path from "{}" to "{}" gives: {}'.format(source, target, nx.has_path(G, source, target)))


paths = []
total_paths = 0
existing_paths = 0
for n1 in G.nodes():
    for n2 in G.nodes():
        if not n1 == n2:
            total_paths += 1
            try:
                shortest_path = nx.shortest_path(G, source=n1, target=n2)
                shortest_path_vulns = []
                for node_id in range(len(shortest_path) - 1):
                    step_source = shortest_path[node_id]
                    step_destination = shortest_path[node_id+1]
                    shortest_path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
                shortest_path_length = nx.shortest_path_length(G, source=n1, target=n2)
                all_paths = list(nx.all_simple_paths(G, source=n1, target=n2))
                all_paths_vulns = []
                for path in all_paths:
                    path_vulns = []
                    for node_id in range(len(path) - 1):
                        step_source = path[node_id]
                        step_destination = path[node_id+1]
                        path_vulns.append(G.get_edge_data(step_source, step_destination)['vulns'])
                    all_paths_vulns.append(path_vulns)
                existing_paths += 1
                paths.append({'source': n1, 'target': n2, 'shortest_path': shortest_path, 'length': shortest_path_length, 'shortest_path_vulns': shortest_path_vulns, 'all_paths': all_paths, 'all_paths_vulns': all_paths_vulns})
            except nx.NetworkXNoPath:
                continue
print('paths: {}'.format(paths[-1]))
print('total_paths: {}'.format(total_paths))
print('existing_paths: {}'.format(existing_paths))

all_paths_dict = dict(nx.all_pairs_shortest_path(G))
print('{}'.format(all_paths_dict['guest@ubu144-192.168.10.19']['user@win7-192.168.10.9']))



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


# G = nx.dodecahedral_graph()
# nx.draw(G)
# plt.show()

# A = nx.nx_agraph.to_agraph(G)

# G = pgv.AGraph('data/ag.graphml')
# A.draw()