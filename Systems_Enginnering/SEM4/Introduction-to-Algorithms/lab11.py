import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from collections import defaultdict, deque
import time
import os

class GraphAlgorithms:
    def __init__(self, filepath):
        self.graph = defaultdict(list)
        self.weighted_edges = []
        self.vertices = set()
        self.pos = None
        self.load_graph(filepath)
        
    def load_graph(self, filepath):
        """Wczytuje graf z pliku"""
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            v1, v2 = parts[0], parts[1]
                            weight = float(parts[2]) if len(parts) > 2 else 1.0
                            
                            self.vertices.add(v1)
                            self.vertices.add(v2)
                            self.graph[v1].append(v2)
                            self.graph[v2].append(v1)  # Nieskierowany
                            self.weighted_edges.append((v1, v2, weight))
            
            print(f"Graf wczytany: {len(self.vertices)} wierzchołków, {len(self.weighted_edges)} krawędzi")
        except FileNotFoundError:
            print(f"Nie można znaleźć pliku: {filepath}")
            
    def create_networkx_graph(self, directed=False):
        """Tworzy graf NetworkX do wizualizacji"""
        G = nx.DiGraph() if directed else nx.Graph()
        for v1, v2, weight in self.weighted_edges:
            G.add_edge(v1, v2, weight=weight)
        return G
        
    def get_layout(self, G):
        """Generuje layout dla grafu"""
        if self.pos is None:
            self.pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        return self.pos

class DFSVisualizer:
    def __init__(self, graph_algo):
        self.graph_algo = graph_algo
        self.visited = set()
        self.stack = []
        self.current = None
        self.steps = []
        
    def dfs_step_by_step(self, start_vertex):
        """DFS krok po kroku"""
        self.visited = set()
        self.stack = [start_vertex]
        self.steps = []
        
        print(f"\n=== ZADANIE 1: DFS z wierzchołka {start_vertex} ===")
        
        step = 0
        while self.stack:
            step += 1
            current = self.stack.pop()
            
            if current not in self.visited:
                self.visited.add(current)
                self.current = current
                
                # Zapisz krok
                step_info = {
                    'step': step,
                    'current': current,
                    'visited': self.visited.copy(),
                    'stack': self.stack.copy(),
                    'action': f'Odwiedzam wierzchołek {current}'
                }
                self.steps.append(step_info)
                
                print(f"Krok {step}: Odwiedzam {current}")
                print(f"  Stos: {self.stack}")
                print(f"  Odwiedzone: {sorted(self.visited)}")
                
                # Dodaj sąsiadów do stosu (w odwrotnej kolejności alfabetycznej)
                neighbors = sorted(self.graph_algo.graph[current], reverse=True)
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.stack.append(neighbor)
                        
        return self.steps
        
    def visualize_dfs(self, start_vertex):
        """Wizualizacja DFS"""
        steps = self.dfs_step_by_step(start_vertex)
        G = self.graph_algo.create_networkx_graph()
        pos = self.graph_algo.get_layout(G)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'DFS - Przeszukiwanie w głąb z wierzchołka {start_vertex}', fontsize=16)
        
        def animate(frame):
            if frame < len(steps):
                step = steps[frame]
                ax1.clear()
                ax2.clear()
                
                # Graf
                ax1.set_title(f"Krok {step['step']}: {step['action']}")
                
                # Rysuj wszystkie krawędzie
                nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='lightgray', width=1)
                
                # Koloruj wierzchołki
                node_colors = []
                for node in G.nodes():
                    if node == step['current']:
                        node_colors.append('red')  # Aktualny
                    elif node in step['visited']:
                        node_colors.append('lightblue')  # Odwiedzony
                    elif node in step['stack']:
                        node_colors.append('yellow')  # Na stosie
                    else:
                        node_colors.append('white')  # Nieodwiedzony
                        
                nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                                     node_size=500, edgecolors='black')
                nx.draw_networkx_labels(G, pos, ax=ax1)
                
                # Informacje o kroku
                ax2.text(0.1, 0.9, f"Krok: {step['step']}", fontsize=12, weight='bold')
                ax2.text(0.1, 0.8, f"Aktualny: {step['current']}", fontsize=11)
                ax2.text(0.1, 0.7, f"Stos: {step['stack']}", fontsize=11)
                ax2.text(0.1, 0.6, f"Odwiedzone: {sorted(step['visited'])}", fontsize=11)
                ax2.text(0.1, 0.4, "Legenda:", fontsize=12, weight='bold')
                ax2.text(0.1, 0.3, "🔴 Aktualny wierzchołek", fontsize=10)
                ax2.text(0.1, 0.25, "🔵 Odwiedzony", fontsize=10)
                ax2.text(0.1, 0.2, "🟡 Na stosie", fontsize=10)
                ax2.text(0.1, 0.15, "⚪ Nieodwiedzony", fontsize=10)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
        
        ani = animation.FuncAnimation(fig, animate, frames=len(steps), 
                                    interval=2000, repeat=True)
        plt.tight_layout()
        plt.show()
        return ani

class FullDFSVisualizer:
    def __init__(self, graph_algo):
        self.graph_algo = graph_algo
        
    def full_dfs_step_by_step(self):
        """Pełne DFS - wszystkie komponenty spójności"""
        visited = set()
        components = []
        steps = []
        step = 0
        
        print(f"\n=== ZADANIE 2: Pełne przeszukiwanie w głąb ===")
        
        for vertex in sorted(self.graph_algo.vertices):
            if vertex not in visited:
                step += 1
                component = []
                stack = [vertex]
                
                print(f"\nKomponent {len(components) + 1} - start z {vertex}:")
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        step_info = {
                            'step': step,
                            'current': current,
                            'visited': visited.copy(),
                            'component': component.copy(),
                            'component_num': len(components) + 1
                        }
                        steps.append(step_info)
                        
                        print(f"  Odwiedzam: {current}")
                        
                        # Dodaj sąsiadów
                        neighbors = sorted(self.graph_algo.graph[current], reverse=True)
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                        step += 1
                
                components.append(component)
                print(f"  Komponent: {component}")
        
        print(f"\nZnaleziono {len(components)} komponent(ów) spójności:")
        for i, comp in enumerate(components, 1):
            print(f"  Komponent {i}: {comp}")
            
        return steps, components
        
    def visualize_full_dfs(self):
        """Wizualizacja pełnego DFS"""
        steps, components = self.full_dfs_step_by_step()
        G = self.graph_algo.create_networkx_graph()
        pos = self.graph_algo.get_layout(G)
        
        # Przypisz kolory komponentom
        component_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Pełne przeszukiwanie w głąb - komponenty spójności', fontsize=16)
        
        def animate(frame):
            if frame < len(steps):
                step = steps[frame]
                ax1.clear()
                ax2.clear()
                
                ax1.set_title(f"Krok {step['step']} - Komponent {step['component_num']}")
                
                # Rysuj krawędzie
                nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='lightgray', width=1)
                
                # Koloruj wierzchołki według komponentów
                node_colors = []
                for node in G.nodes():
                    if node == step['current']:
                        node_colors.append('red')
                    elif node in step['visited']:
                        # Znajdź do którego komponentu należy
                        comp_idx = 0
                        for i, comp in enumerate(components):
                            if node in comp:
                                comp_idx = i
                                break
                        node_colors.append(component_colors[comp_idx % len(component_colors)])
                    else:
                        node_colors.append('white')
                
                nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                                     node_size=500, edgecolors='black')
                nx.draw_networkx_labels(G, pos, ax=ax1)
                
                # Informacje
                ax2.text(0.1, 0.9, f"Krok: {step['step']}", fontsize=12, weight='bold')
                ax2.text(0.1, 0.8, f"Aktualny komponent: {step['component_num']}", fontsize=11)
                ax2.text(0.1, 0.7, f"Aktualny wierzchołek: {step['current']}", fontsize=11)
                ax2.text(0.1, 0.6, f"Komponent: {step['component']}", fontsize=11)
                
                y_pos = 0.4
                ax2.text(0.1, y_pos, "Komponenty:", fontsize=12, weight='bold')
                for i, comp in enumerate(components[:step['component_num']]):
                    y_pos -= 0.05
                    ax2.text(0.1, y_pos, f"  {i+1}: {comp}", fontsize=10)
                
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
        
        ani = animation.FuncAnimation(fig, animate, frames=len(steps),
                                    interval=1500, repeat=True)
        plt.tight_layout()
        plt.show()
        return ani

class DFSEdgeClassifier:
    def __init__(self, graph_algo):
        self.graph_algo = graph_algo
        
    def dfs_with_edge_classification(self):
        """DFS z klasyfikacją krawędzi"""
        visited = set()
        discovery = {}
        finish = {}
        edge_types = {}
        time = [0]  # Lista aby móc modyfikować w zagnieżdżonych funkcjach
        steps = []
        
        print(f"\n=== ZADANIE 3: DFS z klasyfikacją krawędzi ===")
        
        def dfs_visit(u):
            visited.add(u)
            discovery[u] = time[0]
            time[0] += 1
            
            print(f"Odkrywam {u} w czasie {discovery[u]}")
            
            for v in sorted(self.graph_algo.graph[u]):
                edge = (u, v) if u < v else (v, u)
                
                if v not in visited:
                    # Krawędź drzewowa
                    edge_types[edge] = 'tree'
                    print(f"  Krawędź drzewowa: {u} -> {v}")
                    dfs_visit(v)
                else:
                    # Klasyfikuj krawędź
                    if v not in finish:
                        # Krawędź powrotna
                        edge_types[edge] = 'back'
                        print(f"  Krawędź powrotna: {u} -> {v}")
                    elif discovery[u] < discovery[v]:
                        # Krawędź w przód
                        edge_types[edge] = 'forward'
                        print(f"  Krawędź w przód: {u} -> {v}")
                    else:
                        # Krawędź poprzeczna
                        edge_types[edge] = 'cross'
                        print(f"  Krawędź poprzeczna: {u} -> {v}")
            
            finish[u] = time[0]
            time[0] += 1
            print(f"Kończę {u} w czasie {finish[u]}")
            
            steps.append({
                'vertex': u,
                'discovery': discovery.copy(),
                'finish': finish.copy(),
                'edge_types': edge_types.copy(),
                'visited': visited.copy()
            })
        
        # Rozpocznij DFS od każdego nieodwiedzonego wierzchołka
        for vertex in sorted(self.graph_algo.vertices):
            if vertex not in visited:
                dfs_visit(vertex)
        
        # Podsumowanie
        print(f"\nKlasyfikacja krawędzi:")
        edge_counts = {'tree': 0, 'back': 0, 'forward': 0, 'cross': 0}
        for edge, edge_type in edge_types.items():
            print(f"  {edge}: {edge_type}")
            edge_counts[edge_type] += 1
            
        print(f"\nPodsumowanie:")
        for edge_type, count in edge_counts.items():
            print(f"  {edge_type}: {count}")
        
        return steps, edge_types, discovery, finish
        
    def visualize_dfs_edges(self):
        """Wizualizacja DFS z klasyfikacją krawędzi"""
        steps, edge_types, discovery, finish = self.dfs_with_edge_classification()
        G = self.graph_algo.create_networkx_graph()
        pos = self.graph_algo.get_layout(G)
        
        edge_colors = {
            'tree': 'green',
            'back': 'red', 
            'forward': 'orange',
            'cross': 'purple'
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('DFS z klasyfikacją krawędzi', fontsize=16)
        
        # Statyczna wizualizacja końcowego wyniku
        ax1.set_title("Klasyfikacja krawędzi")
        
        # Rysuj krawędzie według typu
        for edge, edge_type in edge_types.items():
            u, v = edge
            color = edge_colors.get(edge_type, 'gray')
            nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax1, 
                                 edge_color=color, width=2)
        
        # Rysuj wierzchołki z czasami odkrycia/zakończenia
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue',
                             node_size=700, edgecolors='black')
        
        # Etykiety z czasami
        labels = {}
        for node in G.nodes():
            labels[node] = f"{node}\n{discovery[node]}/{finish[node]}"
        nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
        
        # Legenda
        ax2.text(0.1, 0.9, "Legenda krawędzi:", fontsize=12, weight='bold')
        y_pos = 0.8
        for edge_type, color in edge_colors.items():
            count = sum(1 for et in edge_types.values() if et == edge_type)
            ax2.plot([0.1, 0.2], [y_pos, y_pos], color=color, linewidth=3)
            ax2.text(0.25, y_pos, f"{edge_type.capitalize()}: {count}", fontsize=11)
            y_pos -= 0.1
            
        ax2.text(0.1, 0.4, "Czas odkrycia/zakończenia:", fontsize=12, weight='bold')
        y_pos = 0.3
        for vertex in sorted(discovery.keys()):
            ax2.text(0.1, y_pos, f"{vertex}: {discovery[vertex]}/{finish[vertex]}", fontsize=10)
            y_pos -= 0.05
            
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

class TopologicalSort:
    def __init__(self, graph_algo):
        self.graph_algo = graph_algo
        
    def topological_sort_dfs(self):
        """Sortowanie topologiczne używając DFS"""
        visited = set()
        stack = []
        steps = []
        
        print(f"\n=== ZADANIE 4: Sortowanie topologiczne ===")
        
        def dfs_visit(vertex):
            visited.add(vertex)
            print(f"Odwiedzam: {vertex}")
            
            for neighbor in sorted(self.graph_algo.graph[vertex]):
                if neighbor not in visited:
                    dfs_visit(neighbor)
            
            stack.append(vertex)
            print(f"Dodaję do stosu: {vertex}")
            
            steps.append({
                'vertex': vertex,
                'visited': visited.copy(),
                'stack': stack.copy()
            })
        
        # Uruchom DFS dla wszystkich nieodwiedzonych wierzchołków
        for vertex in sorted(self.graph_algo.vertices):
            if vertex not in visited:
                dfs_visit(vertex)
        
        # Odwróć stos aby otrzymać porządek topologiczny
        topological_order = stack[::-1]
        
        print(f"\nPorządek topologiczny: {topological_order}")
        return steps, topological_order
        
    def visualize_topological_sort(self):
        """Wizualizacja sortowania topologicznego"""
        steps, topo_order = self.topological_sort_dfs()
        
        # Stwórz graf skierowany dla lepszej wizualizacji
        G = nx.DiGraph()
        for v1, v2, _ in self.graph_algo.weighted_edges:
            G.add_edge(v1, v2)
        
        pos = self.graph_algo.get_layout(G)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Sortowanie topologiczne', fontsize=16)
        
        # Wizualizacja końcowego wyniku
        ax1.set_title("Graf z porządkiem topologicznym")
        
        # Rysuj graf
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', 
                             arrows=True, arrowsize=20, width=1)
        
        # Koloruj wierzchołki według pozycji w porządku topologicznym
        colors = plt.cm.viridis([i/len(topo_order) for i in range(len(topo_order))])
        node_colors = []
        for node in G.nodes():
            if node in topo_order:
                idx = topo_order.index(node)
                node_colors.append(colors[idx])
            else:
                node_colors.append('white')
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                             node_size=700, edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax1)
        
        # Pokaż porządek topologiczny
        ax2.text(0.1, 0.9, "Porządek topologiczny:", fontsize=12, weight='bold')
        ax2.text(0.1, 0.8, " → ".join(topo_order), fontsize=14, weight='bold')
        
        ax2.text(0.1, 0.6, "Pozycje:", fontsize=12, weight='bold')
        y_pos = 0.5
        for i, vertex in enumerate(topo_order):
            ax2.text(0.1, y_pos, f"{i+1}. {vertex}", fontsize=11)
            y_pos -= 0.05
            
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

class UnionFind:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

class KruskalMST:
    def __init__(self, graph_algo):
        self.graph_algo = graph_algo
        
    def kruskal_step_by_step(self):
        """Algorytm Kruskala krok po kroku"""
        edges = sorted(self.graph_algo.weighted_edges, key=lambda x: x[2])
        uf = UnionFind(self.graph_algo.vertices)
        mst = []
        steps = []
        total_weight = 0
        
        print(f"\n=== ZADANIE 5: Algorytm Kruskala ===")
        print(f"Krawędzie posortowane według wagi:")
        for i, (u, v, w) in enumerate(edges):
            print(f"  {i+1}. ({u}, {v}): {w}")
        
        print(f"\nPrzebieg algorytmu:")
        
        for step, (u, v, weight) in enumerate(edges, 1):
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst.append((u, v, weight))
                total_weight += weight
                status = "DODANA do MST"
                print(f"Krok {step}: ({u}, {v}) waga {weight} - {status}")
            else:
                status = "ODRZUCONA (cykl)"
                print(f"Krok {step}: ({u}, {v}) waga {weight} - {status}")
            
            steps.append({
                'step': step,
                'edge': (u, v, weight),
                'status': status,
                'mst': mst.copy(),
                'total_weight': total_weight
            })
        
        print(f"\nMinimalne drzewo rozpinające:")
        for u, v, w in mst:
            print(f"  ({u}, {v}): {w}")
        print(f"Całkowita waga MST: {total_weight}")
        
        return steps, mst, total_weight
        
    def visualize_kruskal(self):
        """Wizualizacja algorytmu Kruskala"""
        steps, mst, total_weight = self.kruskal_step_by_step()
        G = self.graph_algo.create_networkx_graph()
        pos = self.graph_algo.get_layout(G)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Algorytm Kruskala - Minimalne drzewo rozpinające', fontsize=16)
        
        def animate(frame):
            if frame < len(steps):
                step = steps[frame]
                ax1.clear()
                ax2.clear()
                
                ax1.set_title(f"Krok {step['step']}: {step['status']}")
                
                # Rysuj wszystkie krawędzie jako szare
                for u, v, w in self.graph_algo.weighted_edges:
                    nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax1,
                                         edge_color='lightgray', width=1)
                
                # Rysuj krawędzie MST jako zielone
                mst_edges = [(u, v) for u, v, w in step['mst']]
                if mst_edges:
                    nx.draw_networkx_edges(G, pos, mst_edges, ax=ax1,
                                         edge_color='green', width=3)
                
                # Podświetl aktualnie sprawdzaną krawędź
                current_edge = step['edge']
                color = 'green' if step['status'] == "DODANA do MST" else 'red'
                nx.draw_networkx_edges(G, pos, [(current_edge[0], current_edge[1])], 
                                     ax=ax1, edge_color=color, width=2)
                
                # Rysuj wierzchołki
                nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue',
                                     node_size=500, edgecolors='black')
                nx.draw_networkx_labels(G, pos, ax=ax1)
                
                # Rysuj wagi krawędzi
                edge_labels = {}
                for u, v, w in self.graph_algo.weighted_edges:
                    edge_labels[(u, v)] = str(w)
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
                
                # Informacje o kroku
                ax2.text(0.1, 0.9, f"Krok: {step['step']}", fontsize=12, weight='bold')
                ax2.text(0.1, 0.85, f"Sprawdzana krawędź: ({current_edge[0]}, {current_edge[1]})", fontsize=11)
                ax2.text(0.1, 0.8, f"Waga: {current_edge[2]}", fontsize=11)
                ax2.text(0.1, 0.75, f"Status: {step['status']}", fontsize=11)
                ax2.text(0.1, 0.65, f"Całkowita waga MST: {step['total_weight']}", fontsize=11)
                
                ax2.text(0.1, 0.55, "Krawędzie w MST:", fontsize=12, weight='bold')
                y_pos = 0.5
                for u, v, w in step['mst']:
                    ax2.text(0.1, y_pos, f"  ({u}, {v}): {w}", fontsize=10)
                    y_pos -= 0.04
                
                ax2.text(0.1, 0.2, "Legenda:", fontsize=12, weight='bold')
                ax2.text(0.1, 0.15, "🟢 Krawędź w MST", fontsize=10)
                ax2.text(0.1, 0.1, "🔴 Krawędź odrzucona", fontsize=10)
                ax2.text(0.1, 0.05, "⚪ Krawędź niesprawd.", fontsize=10)
                
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
        
        ani = animation.FuncAnimation(fig, animate, frames=len(steps),
                                    interval=2000, repeat=True)
        plt.tight_layout()
        plt.show()
        return ani

def create_sample_graph():
    """Tworzy przykładowy plik z grafem"""
    sample_content = """# Graph format: vertex1 vertex2 weight
# Example graph with 6 vertices (A, B, C, D, E, F)
A B 4
A C 2
B C 1
B D 5
C D 8
C E 10
D E 2
D F 6
E F 3
B E 7"""
    
    with open('graph.txt', 'w') as f:
        f.write(sample_content)
    print("Utworzono przykładowy plik graph.txt")

def zadanie1(path):
    """Zadanie 1: DFS z wizualizacją"""
    print("="*50)
    print("ZADANIE 1: Przeszukiwanie w głąb (DFS)")
    print("="*50)
    
    graph_algo = GraphAlgorithms(path)
    if not graph_algo.vertices:
        print("Błąd: Nie udało się wczytać grafu")
        return
    
    dfs_viz = DFSVisualizer(graph_algo)
    
    # Wybierz pierwszy wierzchołek jako startowy
    start_vertex = sorted(graph_algo.vertices)[0]
    print(f"Wierzchołek startowy: {start_vertex}")
    
    # Wykonaj DFS i pokaż wizualizację
    ani = dfs_viz.visualize_dfs(start_vertex)
    return ani

def zadanie2(path):
    """Zadanie 2: Pełne DFS"""
    print("="*50)
    print("ZADANIE 2: Pełne przeszukiwanie w głąb")
    print("="*50)
    
    graph_algo = GraphAlgorithms(path)
    if not graph_algo.vertices:
        print("Błąd: Nie udało się wczytać grafu")
        return
    
    full_dfs_viz = FullDFSVisualizer(graph_algo)
    ani = full_dfs_viz.visualize_full_dfs()
    return ani

def zadanie3(path):
    """Zadanie 3: DFS z klasyfikacją krawędzi"""
    print("="*50)
    print("ZADANIE 3: DFS z klasyfikacją krawędzi")
    print("="*50)
    
    graph_algo = GraphAlgorithms(path)
    if not graph_algo.vertices:
        print("Błąd: Nie udało się wczytać grafu")
        return
    
    dfs_edges = DFSEdgeClassifier(graph_algo)
    dfs_edges.visualize_dfs_edges()

def zadanie4(path):
    """Zadanie 4: Sortowanie topologiczne"""
    print("="*50)
    print("ZADANIE 4: Sortowanie topologiczne")
    print("="*50)
    
    graph_algo = GraphAlgorithms(path)
    if not graph_algo.vertices:
        print("Błąd: Nie udało się wczytać grafu")
        return
    
    topo_sort = TopologicalSort(graph_algo)
    topo_sort.visualize_topological_sort()

def zadanie5(path):
    """Zadanie 5: Algorytm Kruskala"""
    print("="*50)
    print("ZADANIE 5: Minimalne drzewo rozpinające (Kruskal)")
    print("="*50)
    
    graph_algo = GraphAlgorithms(path)
    if not graph_algo.vertices:
        print("Błąd: Nie udało się wczytać grafu")
        return
    
    kruskal = KruskalMST(graph_algo)
    ani = kruskal.visualize_kruskal()
    return ani

if __name__ == "__main__":
                path = "graph.txt"
                zadanie1(path)
                zadanie2(path)
                zadanie3(path)
                zadanie4(path)
                zadanie5(path)
