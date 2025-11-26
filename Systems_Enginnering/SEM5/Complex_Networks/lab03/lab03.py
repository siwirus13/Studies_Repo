
import networkx as nx
import numpy as np

# =====================================================
# === SAFE EDGE LIST LOADER (handles extra columns) ===
# =====================================================
def load_edge_file(filename, directed=False):
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


# =====================================================
# === ANALIZA SIECI (POLSKIE OPISY PARAMETRÓW) ========
# =====================================================
def task_lab03(G, name="Graf"):
    print(f"\n=== Analiza sieci: {name} ===")

    # --- Podstawowe miary ---
    n = G.number_of_nodes()      # liczba wierzchołków
    m = G.number_of_edges()      # liczba krawędzi
    degrees = [deg for _, deg in G.degree()]
    avg_degree = np.mean(degrees)
    min_degree = np.min(degrees)
    max_degree = np.max(degrees)
    density = nx.density(G)

    # --- Spójność ---
    try:
        if nx.is_connected(G):
            avg_shortest_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_shortest_path = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
    except Exception:
        avg_shortest_path = None
        diameter = None

    # --- Współczynnik skupienia ---
    clustering_coeff = nx.average_clustering(G)

    # --- Centralności ---
    deg_centrality = nx.degree_centrality(G)
    close_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.NetworkXException:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        eigen_sub = nx.eigenvector_centrality_numpy(subgraph)
        eigenvector_centrality = {n: eigen_sub.get(n, 0.0) for n in G.nodes()}

    # --- Składowe ---
    num_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len))

    # --- Mosty i punkty artykulacji ---
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))

    # === WYDRUK WYNIKÓW (POLSKIE OPISY) ===
    print("=" * 70)
    print("PODSTAWOWE PARAMETRY GRAFU")
    print("=" * 70)
    print(f"Rozmiar (liczba wierzchołków): {n}")
    print(f"Liczba połączeń (krawędzi): {m}")
    print(f"Średni stopień (rząd grafu): {avg_degree:.3f}")
    print(f"Minimalny stopień: {min_degree}, Maksymalny stopień: {max_degree}")
    print(f"Gęstość (stopień wypełnienia grafu): {density:.5f}")

    if avg_shortest_path is not None:
        print(f"Średnia długość ścieżki: {avg_shortest_path:.3f}")
        print(f"Średnica (najgorsze skomunikowanie): {diameter}")
    else:
        print("Średnia długość ścieżki: N/D (graf niespójny)")
        print("Średnica: N/D (graf niespójny)")

    print(f"Średni współczynnik skupienia: {clustering_coeff:.5f}")
    print(f"Liczba składowych spójnych: {num_components}")
    print(f"Rozmiar największej składowej: {largest_cc_size}")
    print(f"Liczba punktów artykulacji: {len(articulation_points)}")
    print(f"Liczba mostów: {len(bridges)}")

    # === Centralności (polskie opisy) ===
    print("\n" + "=" * 70)
    print("TOP 5 – Stopień węzła (liczba połączeń)")
    print("=" * 70)
    for node, val in sorted(deg_centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"  Węzeł {node}: {val:.3f} (stopień: {G.degree(node)})")

    print("\n" + "=" * 70)
    print("TOP 5 – Bliskość (Closeness)")
    print("Odwrotność średniej odległości od innych węzłów")
    print("=" * 70)
    for node, val in sorted(close_centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"  Węzeł {node}: {val:.3f}")

    print("\n" + "=" * 70)
    print("TOP 5 – Pośrednictwo (Betweenness)")
    print("Odsetek najkrótszych ścieżek przechodzących przez węzeł")
    print("=" * 70)
    for node, val in sorted(betweenness_centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"  Węzeł {node}: {val:.3f}")

    print("\n" + "=" * 70)
    print("TOP 5 – Centralność wektorowa (Eigenvector)")
    print("(dla największej składowej grafu)")
    print("=" * 70)
    top_eigen = sorted(eigenvector_centrality.items(), key=lambda x: -x[1])[:5]
    if top_eigen[0][1] > 0:
        for node, val in top_eigen:
            print(f"  Węzeł {node}: {val:.3f}")
    else:
        print("  Brak znaczących wartości (graf niespójny)")

    print("\nAnaliza zakończona.\n")


# =====================================================
# === GŁÓWNY PROGRAM ==================================
# =====================================================
def main():
    print("Wczytywanie plików z danymi...\n")

    bio_graph = load_edge_file("data/bio-DM-LC.edges", directed=False)
    mammalia_graph = load_edge_file("data/mammalia-voles-bhp-trapping-28.edges", directed=False)

    task_lab03(bio_graph, "bio-DM-LC")
    task_lab03(mammalia_graph, "mammalia-voles-bhp-trapping")


# =====================================================
# === URUCHOMIENIE ====================================
# =====================================================
if __name__ == "__main__":
    main()
