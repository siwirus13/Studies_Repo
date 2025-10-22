import networkx as nx


graph = nx.read_edgelist("data/bio-DM-LC.edges", data=(("weight", float),))

def task1(G):

    print("Program will try to seek shortest path between two nodes")
    first_node = input("Input first node")
    second_node = input("Input second node")

    try:
        path = nx.shortest_path(G, source = first_node, target = second_node)

        print(f"Shortest path starting from node {first_node} ", path)
        print("Lenght:", len(path) -1)

    except nx.NetworkXNoPath:
        print("No path found between those nodes.")
    except nx.NodeNotFound as e:
        print(e)

def task2(G):
    if nx.is_eulerian(G):
         print("The graph has an Eulerian circuit.")
    elif nx.has_eulerian_path(G):
        print("The graph has an Eulerian path (but not a full circuit).")
    else:
        print("The graph does not have an Eulerian path.")


def task3(G):

    print("Program will try to maximize flow between two nodes")
    first_node = input("Input first node")
    second_node = input("Input second node")
    flow_value, flow_dict = nx.maximum_flow(G, first_node, second_node, capacity='weight')
    print(f"Maximum flow value from {first_node} to {second_node}: {flow_value}")
   

task1(graph)
task2(graph)
task3(graph)
