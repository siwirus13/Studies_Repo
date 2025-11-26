
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd

# =====================================================
# === TASK 2 — Graph analysis and visualization =======
# =====================================================

def load_edge_file(filename, directed=False):
    """
    Safely load an edge list (ignores extra columns, comments, and empty lines).
    """
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('%') or not line.strip():
                continue  # skip comments or empty lines
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                edges.append((u, v))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    return G


def task2():
    print("Program will analyze and visualize the graph: mammalia-voles-bhp-trapping")

    # === Load graph safely ===
    filename = "data/mammalia-voles-bhp-trapping-28.edges"
    G = load_edge_file(filename, directed=False)

    print(f"\nGraph loaded successfully!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # === Compute basic properties ===
    reciprocal_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    degrees = [G.degree(n) for n in G.nodes()]
    clustering = nx.average_clustering(G)
    pagerank = nx.pagerank(G)
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n=== BASIC GRAPH STATS ===")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Min degree: {np.min(degrees)}, Max degree: {np.max(degrees)}")
    print(f"Average clustering coefficient: {clustering:.4f}")
    print(f"Reciprocal edges (if any): {reciprocal_edges}")

    print("\n=== TOP 10 NODES BY PAGERANK ===")
    for node, score in top_nodes:
        print(f"Node {node}: {score:.4f} (degree={G.degree(node)})")

    # === Generate and save adjacency and incidence matrices ===
    print("\nGenerating adjacency and incidence matrices...")

    # Macierz sąsiedztwa
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    nodes = sorted(G.nodes())
    adjacency_df = pd.DataFrame(A, index=nodes, columns=nodes)
    adjacency_df.to_csv("mammalia_voles_adjacency_matrix.csv")
    print("Adjacency matrix saved to mammalia_voles_adjacency_matrix.csv")

    # Macierz incydencji
    # Uwaga: dla grafu nieskierowanego -> kolumny to krawędzie, wiersze to wierzchołki
    B = nx.incidence_matrix(G, oriented=True).toarray()
    edges = [f"{u}-{v}" for u, v in G.edges()]
    incidence_df = pd.DataFrame(B, index=nodes, columns=edges)
    incidence_df.to_csv("mammalia_voles_incidence_matrix.csv")
    print("Incidence matrix saved to mammalia_voles_incidence_matrix.csv")

    # === Matplotlib visualization ===
    print("\nGenerating static visualization (matplotlib)...")
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=80,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.85,
        edgecolors='black',
        linewidths=0.5
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.6)
    plt.title("Mammalia-Voles-BHP-Trapping Network", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("mammalia_voles_network.png", dpi=300)
    plt.close()

    # === Interactive visualization (PyVis) ===
    print("Generating interactive visualization (PyVis)...")
    net = Network(
        notebook=False,
        width='1200px',
        height='900px',
        bgcolor='#ffffff',
        font_color='#000000',
        directed=False
    )

    for node in G.nodes():
        deg = G.degree(node)
        net.add_node(
            node,
            label=str(node),
            title=f"Node {node}\nDegree: {deg}",
            size=5 + deg * 2,
            color='#ff6b6b' if deg > 5 else '#4ecdc4'
        )

    for u, v in G.edges():
        net.add_edge(u, v)

    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "font": {"size": 12, "face": "arial", "bold": true}
      },
      "edges": {
        "color": {"color": "#666666", "highlight": "#ff0000"},
        "smooth": {"enabled": true, "type": "curvedCW", "roundness": 0.2},
        "width": 1.5
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.05
        }
      },
      "interaction": {"hover": true, "navigationButtons": true}
    }
    """)

    net.write_html("mammalia_voles_network.html")
    print("Visualizations saved: mammalia_voles_network.png and mammalia_voles_network.html")


# =====================================================
# === MAIN EXECUTION ==================================
# =====================================================

if __name__ == "__main__":
    task2()

