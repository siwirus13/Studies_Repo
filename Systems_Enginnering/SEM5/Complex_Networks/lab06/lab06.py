import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
from pyvis.network import Network
import warnings
import sys

# Konfiguracja
FILE_PATH = 'Wiki-Vote.txt'
warnings.filterwarnings("ignore") # Wyłączenie ostrzeżeń dla czytelności konsoli

def full_social_network_analysis(file_path):
    print(f"--- 1. WCZYTYWANIE DANYCH: {file_path} ---")
    try:
        # Wczytujemy jako graf skierowany (DiGraph)
        G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, comments='#')
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{file_path}'. Upewnij się, że jest w folderze ze skryptem.")
        return

    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f"Liczba węzłów: {N}")
    print(f"Liczba krawędzi: {E}")

    # --- CZĘŚĆ A: STATYSTYKI I HUBY ---
    print("\n--- 2. ANALIZA PODSTAWOWA I HUBY ---")
    
    # Gęstość i Wzajemność
    density = nx.density(G)
    reciprocity = nx.reciprocity(G)
    print(f"Gęstość sieci: {density:.6f}")
    print(f"Wzajemność (Reciprocity): {reciprocity:.4f} ({reciprocity*100:.2f}%)")

    # In-Degree (Kandydaci - Prestiż)
    in_degrees = dict(G.in_degree())
    sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    print("\nTOP 5 - Najwięcej głosów otrzymanych (Kandydaci/Prestiż):")
    for node, count in sorted_in[:5]:
        print(f" -> Użytkownik {node}: {count} głosów")

    # Out-Degree (Głosujący - Aktywność)
    out_degrees = dict(G.out_degree())
    sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    print("\nTOP 5 - Najwięcej głosów oddanych (Aktywiści):")
    for node, count in sorted_out[:5]:
        print(f" -> Użytkownik {node}: oddał {count} głosów")

    # Betweenness Centrality (Mosty)
    print("\nObliczanie Betweenness Centrality (przybliżenie k=400)...")
    betweenness = nx.betweenness_centrality(G, k=400, seed=42)
    sorted_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    print("TOP 5 - Mosty informacyjne (Betweenness):")
    for node, score in sorted_bet[:5]:
        print(f" -> Użytkownik {node}: {score:.5f}")

    # --- CZĘŚĆ B: WŁAŚCIWOŚCI GRAFU (EULER, ŚCIEŻKI) ---
    print("\n--- 3. WŁAŚCIWOŚCI MATEMATYCZNE ---")
    
    # Euler (z obsługą starszych wersji NetworkX)
    is_eulerian = nx.is_eulerian(G)
    try:
        if hasattr(nx, 'is_semi_eulerian'):
            is_semi = nx.is_semi_eulerian(G)
        elif hasattr(nx, 'has_eulerian_path'):
            is_semi = nx.has_eulerian_path(G)
        else:
            is_semi = False
    except:
        is_semi = False

    print(f"Czy graf jest Eulerowski? {is_eulerian}")
    print(f"Czy graf jest pół-Eulerowski? {is_semi}")

    # Najkrótsza ścieżka i przepływ między HUBAMI
    top_voter = sorted_out[0][0]     # Najaktywniejszy głosujący
    top_candidate = sorted_in[0][0]  # Najpopularniejszy kandydat
    
    print(f"\nBadanie relacji: Top Głosujący ({top_voter}) -> Top Kandydat ({top_candidate})")
    try:
        path = nx.shortest_path(G, source=top_voter, target=top_candidate)
        print(f"Najkrótsza ścieżka: {path}")
        print(f"Długość ścieżki: {len(path)-1} kroków")
        
        # Przepływ
        for u, v in G.edges(): G[u][v]['capacity'] = 1 # Ustawienie przepustowości
        flow_val, _ = nx.maximum_flow(G, top_voter, top_candidate)
        print(f"Maksymalny przepływ (niezależne ścieżki): {flow_val}")
    except nx.NetworkXNoPath:
        print("Brak ścieżki między tymi węzłami.")

    # --- CZĘŚĆ C: WIZUALIZACJE ---
    print("\n--- 4. GENEROWANIE WIZUALIZACJI ---")
    
    # Przygotowanie Rdzenia (K-Core) dla czytelności wykresów
    # Usuwamy szum, zostawiamy tylko "elitę" (min. 15 połączeń)
    k_val = 15
    G_core = nx.k_core(G, k=k_val)
    print(f"Generowanie wykresów dla rdzenia sieci (k={k_val}, węzłów: {G_core.number_of_nodes()})...")

    # 1. MACIERZ SĄSIEDZTWA (SPY PLOT)
    plt.figure(figsize=(8, 8))
    adj_matrix = nx.to_scipy_sparse_array(G_core)
    plt.spy(adj_matrix, markersize=1, color='black')
    plt.title("Wizualizacja Macierzy Sąsiedztwa (Spy Plot)")
    plt.savefig("macierz_sasiedztwa.png", dpi=300)
    plt.close()
    print("[1/4] Zapisano: macierz_sasiedztwa.png")

    # 2. GRAF NETWORKX (STATYCZNY)
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G_core, seed=42, k=0.25)
    node_sizes = [G_core.in_degree(n) * 3 for n in G_core.nodes()] # Rozmiar od popularności
    nx.draw_networkx_nodes(G_core, pos, node_size=node_sizes, node_color='#4a90e2', alpha=0.8)
    nx.draw_networkx_edges(G_core, pos, alpha=0.05, edge_color='gray')
    # Podpisy tylko dla największych
    top_core = sorted(G_core.nodes(), key=lambda n: G_core.in_degree(n), reverse=True)[:10]
    labels = {n: str(n) for n in top_core}
    nx.draw_networkx_labels(G_core, pos, labels, font_size=10, font_weight='bold')
    plt.title(f"Rdzeń Sieci Wiki-Vote (k-core={k_val})")
    plt.axis('off')
    plt.savefig("graf_networkx.png", dpi=300)
    plt.close()
    print("[2/4] Zapisano: graf_networkx.png")

    # 3. GRAF PYVIS (INTERAKTYWNY)
    try:
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', select_menu=True)
        # PyVis nie zawsze lubi wszystkie atrybuty NetworkX, konwersja podstawowa
        net.from_nx(G_core)
        net.barnes_hut()
        net.save_graph("graf_interaktywny.html")
        print("[3/4] Zapisano: graf_interaktywny.html (Otwórz w przeglądarce i zrób screenshot!)")
    except Exception as e:
        print(f"[3/4] Błąd PyVis: {e}")

    # 4. GRAF DWUDZIELNY (Głosujący vs Kandydaci)
    # Tworzymy sztuczny podgraf dla wizualizacji
    top_candidates_list = [n for n, _ in sorted_in[:10]]
    predecessors = set()
    for cand in top_candidates_list:
        predecessors.update(G.predecessors(cand))
    
    # Wybieramy głosujących, którzy nie są w topce kandydatów
    pure_voters = list(predecessors - set(top_candidates_list))[:25] # Bierzemy 25 próbek
    
    B = nx.Graph()
    B.add_nodes_from(pure_voters, bipartite=0)
    B.add_nodes_from(top_candidates_list, bipartite=1)
    for v in pure_voters:
        for c in top_candidates_list:
            if G.has_edge(v, c):
                B.add_edge(v, c)
    
    plt.figure(figsize=(10, 8))
    pos_b = nx.bipartite_layout(B, pure_voters)
    nx.draw_networkx_nodes(B, pos_b, nodelist=pure_voters, node_color='blue', label='Wyborcy', node_size=100)
    nx.draw_networkx_nodes(B, pos_b, nodelist=top_candidates_list, node_color='red', label='Kandydaci', node_size=300)
    nx.draw_networkx_edges(B, pos_b, alpha=0.3)
    nx.draw_networkx_labels(B, pos_b, {n:str(n) for n in top_candidates_list})
    plt.legend()
    plt.title("Struktura Dwudzielna: Wyborcy vs Kandydaci")
    plt.axis('off')
    plt.savefig("graf_dwudzielny.png", dpi=300)
    plt.close()
    print("[4/4] Zapisano: graf_dwudzielny.png")

    print("\n--- KONIEC ANALIZY. DANE GOTOWE DO RAPORTU! ---")

if __name__ == "__main__":
    full_social_network_analysis(FILE_PATH)
