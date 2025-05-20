import numpy as np

from collections import defaultdict

def __dfs(start_node, neighbors, distance = None, parent = None):
    """dfs, probably unnecessary for this application"""
    if not distance:
        distance = {start_node : 0}
        parent = {start_node : None}
        child_distance = 0
    else:
        child_distance = distance[start_node] + 1
    for nbr in neighbors[start_node]:
        if not nbr in parent or distance[nbr] > child_distance:
            parent[nbr] = start_node
            distance[nbr] = child_distance
            __dfs(nbr, neighbors, distance, parent)
    return distance, parent

def __bfs(start_node, neighbors):
    """bfs, used to estimate all distances"""
    to_visit = [start_node]
    parent = {start_node : None}
    distance = {start_node : 0}
    
    while len(to_visit) > 0:
        node = to_visit.pop(0)
        for nbr in neighbors[node]:
            if not nbr in parent:
                to_visit.append(nbr)
                parent[nbr] = node
                distance[nbr] = distance[node] + 1
    
    return distance, parent

def estimate_distances_bfs(nodes, neighbors):
    """
    estimates distances between bays' access points

    nodes: list(tuple)
    neigbors: dict, with all nodes: [neighbor nodes]
    """
    
    graph_distance = np.zeros((len(nodes), len(nodes)))

    for i in range(len(nodes)):
        distance, _ = __bfs(nodes[i], neighbors)
        # assuming undirected graph
        for j in range(i+1, len(nodes)):
            graph_distance[i][j] = distance[nodes[j]]
            graph_distance[j][i] = distance[nodes[j]]
    
    return graph_distance

def edges_to_neighbors(edges):
    """
    turns edges of an undirected graph into a "neighbors" dictionary
    """
    neighbors = defaultdict(list)
    for edge in edges:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    return dict(neighbors)