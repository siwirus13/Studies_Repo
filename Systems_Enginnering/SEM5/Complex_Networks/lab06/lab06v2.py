import networkx as nx
import sys

def calculate_table2_stats(file_path):
    print("--- OBLICZANIE DANYCH DO TABELI 2 ---")
    try:
        # Wczytanie jako skierowany
        G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, comments='#')
        
        # Wersja nieskierowana (do mostów i składowych spójnych)
        G_undir = G.to_undirected()
    except FileNotFoundError:
        print("Brak pliku!")
        return

    # 1. Stopnie (Degrees) - suma in_degree + out_degree
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    
    # 2. Ścieżki i Średnica (liczone na Największej Silnie Spójnej Składowej - LSCC)
    # Ponieważ graf nie jest spójny, średnica = nieskończoność. Liczymy dla rdzenia.
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    
    print("Liczenie średniej ścieżki (to może chwilę potrwać)...")
    try:
        avg_path = nx.average_shortest_path_length(G_sub)
        diameter = nx.diameter(G_sub)
    except:
        avg_path = "N/A (Graf zbyt duży)"
        diameter = "N/A"

    # 3. Klastrowanie
    avg_clustering = nx.average_clustering(G)

    # 4. Struktura (na wersji nieskierowanej)
    # Mosty i punkty artykulacji liczymy na wersji Undirected
    print("Liczenie mostów i punktów artykulacji...")
    bridges = len(list(nx.bridges(G_undir)))
    articulation = len(list(nx.articulation_points(G_undir)))
    
    # Składowe spójne (słabo spójne = traktujemy graf jak nieskierowany)
    num_weak_components = nx.number_weakly_connected_components(G)
    largest_weak_size = len(max(nx.weakly_connected_components(G), key=len))

    print("\n=== WYNIKI DO TABELI 2 ===")
    print(f"Średni stopień: {avg_degree:.3f}")
    print(f"Minimalny stopień: {min(degrees)}")
    print(f"Maksymalny stopień: {max(degrees)}")
    print(f"Średnia długość ścieżki (LSCC): {avg_path:.3f}")
    print(f"Średnica grafu (LSCC): {diameter}")
    print(f"Średni współczynnik skupienia: {avg_clustering:.5f}")
    print(f"Liczba składowych spójnych (Weakly): {num_weak_components}")
    print(f"Rozmiar największej składowej (Weakly): {largest_weak_size}")
    print(f"Liczba punktów artykulacji (Undir): {articulation}")
    print(f"Liczba mostów (Undir): {bridges}")

calculate_table2_stats('Wiki-Vote.txt')
