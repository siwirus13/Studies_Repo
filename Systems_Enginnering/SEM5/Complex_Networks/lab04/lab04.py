import networkx as nx
import matplotlib.pyplot as plt

def task1():
    students = ["A", "B", "C"]
    courses = ["Python", "Databases", "Algebra"]

    edges = [
        ("A", "Python"),
        ("A", "Databases"),
        ("B", "Python"),
        ("B", "Algebra"),
        ("C", "Python"),
        ("C", "Databases"),
        ("C", "Algebra")
    ]

    B = nx.Graph()
    B.add_nodes_from(students, bipartite=0)
    B.add_nodes_from(courses, bipartite=1)
    B.add_edges_from(edges)

    plt.figure(figsize=(8,4))
    pos = {}
    pos.update((node, (0, i)) for i, node in enumerate(students))
    pos.update((node, (1, i)) for i, node in enumerate(courses))
    nx.draw(B, pos, with_labels=True, node_color="skyblue", node_size=1500)
    plt.title("TASK 1 — Bipartite Graph: Students — Courses")
    plt.show()

    P_simple = nx.bipartite.projected_graph(B, students)
    plt.figure(figsize=(5,5))
    nx.draw(P_simple, with_labels=True, node_color="lightgreen", node_size=1500)
    plt.title("TASK 1 — Simple Projection (Students)")
    plt.show()

    P_weighted = nx.bipartite.weighted_projected_graph(B, students)
    plt.figure(figsize=(5,5))
    posw = nx.spring_layout(P_weighted)
    edge_labels = nx.get_edge_attributes(P_weighted, "weight")
    nx.draw(P_weighted, posw, with_labels=True, node_color="orange", node_size=1500)
    nx.draw_networkx_edge_labels(P_weighted, posw, edge_labels=edge_labels)
    plt.title("TASK 1 — Weighted Projection (Students)")
    plt.show()

    P_jaccard = nx.Graph()
    P_jaccard.add_nodes_from(students)

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            s1, s2 = students[i], students[j]
            nbrs1 = set(B.neighbors(s1))
            nbrs2 = set(B.neighbors(s2))
            inter = len(nbrs1 & nbrs2)
            union = len(nbrs1 | nbrs2)
            jaccard = inter / union
            P_jaccard.add_edge(s1, s2, weight=round(jaccard, 2))

    plt.figure(figsize=(5,5))
    pos_j = nx.spring_layout(P_jaccard)
    edge_labels = nx.get_edge_attributes(P_jaccard, "weight")
    nx.draw(P_jaccard, pos_j, with_labels=True, node_color="lightblue", node_size=1500)
    nx.draw_networkx_edge_labels(P_jaccard, pos_j, edge_labels=edge_labels)
    plt.title("TASK 1 — Jaccard Projection (Students)")
    plt.show()


def task2():
    levels = ["1", "2", "3", "4", "5", "6", "7"]
    suits = ["Clubs", "Diamonds", "Hearts", "Spades", "NT"]

    edges = [(l, s) for l in levels for s in suits]

    B = nx.Graph()
    B.add_nodes_from(levels, bipartite=0)
    B.add_nodes_from(suits, bipartite=1)
    B.add_edges_from(edges)

    plt.figure(figsize=(9,5))
    pos = {}
    pos.update((node, (0, i)) for i, node in enumerate(levels))
    pos.update((node, (1, i)) for i, node in enumerate(suits))
    nx.draw(B, pos, with_labels=True, node_color="lightyellow", node_size=1500)
    plt.title("TASK 2 — Bipartite Graph: Level — Suit")
    plt.show()

    P_simple = nx.bipartite.projected_graph(B, levels)
    plt.figure(figsize=(5,5))
    nx.draw(P_simple, with_labels=True, node_color="lightgreen", node_size=1500)
    plt.title("TASK 2 — Simple Projection (Levels)")
    plt.show()

    P_weighted = nx.bipartite.weighted_projected_graph(B, levels)
    plt.figure(figsize=(5,5))
    posw = nx.spring_layout(P_weighted)
    edge_labels = nx.get_edge_attributes(P_weighted, "weight")
    nx.draw(P_weighted, posw, with_labels=True, node_color="orange", node_size=1500)
    nx.draw_networkx_edge_labels(P_weighted, posw, edge_labels=edge_labels)
    plt.title("TASK 2 — Weighted Projection (Levels)")
    plt.show()

    P_jaccard = nx.Graph()
    P_jaccard.add_nodes_from(levels)

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            l1, l2 = levels[i], levels[j]
            nbrs1 = set(B.neighbors(l1))
            nbrs2 = set(B.neighbors(l2))
            inter = len(nbrs1 & nbrs2)
            union = len(nbrs1 | nbrs2)
            jaccard = inter / union
            P_jaccard.add_edge(l1, l2, weight=round(jaccard, 2))

    plt.figure(figsize=(5,5))
    pos_j = nx.spring_layout(P_jaccard)
    edge_labels = nx.get_edge_attributes(P_jaccard, "weight")
    nx.draw(P_jaccard, pos_j, with_labels=True, node_color="lightblue", node_size=1500)
    nx.draw_networkx_edge_labels(P_jaccard, pos_j, edge_labels=edge_labels)
    plt.title("TASK 2 — Jaccard Projection (Levels)")
    plt.show()


if __name__ == "__main__":
    task1()
    task2()
