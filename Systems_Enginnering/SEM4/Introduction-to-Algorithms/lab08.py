#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import deque


class Robot:

    TYPY = ["AGV", "AFV", "ASV", "AUV"]
    
    def __init__(self, typ: str, cena: float, zasieg: int, kamera: int):
        self.typ = typ
        self.cena = cena
        self.zasieg = zasieg
        self.kamera = kamera
    
    def __str__(self):
        return f"Robot(typ={self.typ}, cena={self.cena:.2f}, zasięg={self.zasieg}, kamera={'TAK' if self.kamera else 'NIE'})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            'typ': self.typ,
            'cena': self.cena,
            'zasieg': self.zasieg,
            'kamera': self.kamera
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['typ'], data['cena'], data['zasieg'], data['kamera'])


class RobotStack:
    
    def __init__(self):
        self.items = []
    
    def push(self, robot: Robot):
        self.items.append(robot)
        print(f"Dodano do stosu: {robot}")
    
    def pop(self) -> Optional[Robot]:
        if self.is_empty():
            print("Stos jest pusty!")
            return None
        robot = self.items.pop()
        print(f"Usunięto ze stosu: {robot}")
        return robot
    
    def clear(self):
        print("Czyszczenie stosu:")
        while not self.is_empty():
            self.pop()
        print("Stos został wyczyszczony.")
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)


class RobotQueue:
    
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, robot: Robot):
        self.items.append(robot)
        print(f"Dodano do kolejki: {robot}")
    
    def dequeue(self) -> Optional[Robot]:
        if self.is_empty():
            print("Kolejka jest pusta!")
            return None
        robot = self.items.popleft()
        print(f"Usunięto z kolejki: {robot}")
        return robot
    
    def clear(self):
        print("Czyszczenie kolejki:")
        while not self.is_empty():
            self.dequeue()
        print("Kolejka została wyczyszczona.")
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)


class LinkedListNode:
    
    def __init__(self, robot: Robot, key: str):
        self.robot = robot
        self.key = key
        self.next = None


class RobotLinkedList:
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.nodes = [None] * capacity  
        self.free_indices = list(range(capacity))
        self.head_indices = {}  
        self.size = 0
    
    def _generate_key(self, robot: Robot) -> str:
        return f"{robot.typ}_{robot.cena:.0f}_{robot.zasieg}"
    
    def add(self, robot: Robot):
        if self.size >= self.capacity:
            print("Lista jest pełna!")
            return False
        
        key = self._generate_key(robot)
        index = self.free_indices.pop(0)
        
        node = LinkedListNode(robot, key)
        self.nodes[index] = node
        
        if robot.typ in self.head_indices:
            node.next = self.head_indices[robot.typ]
        self.head_indices[robot.typ] = index
        
        self.size += 1
        print(f"Dodano do listy: {robot} (klucz: {key})")
        return True
    
    def remove(self, key: str) -> bool:
        for typ, head_idx in self.head_indices.items():
            prev_idx = None
            current_idx = head_idx
            
            while current_idx is not None:
                current_node = self.nodes[current_idx]
                if current_node.key == key:
                    if prev_idx is None:
                        self.head_indices[typ] = current_node.next
                    else:
                        self.nodes[prev_idx].next = current_node.next
                    
                    print(f"Usunięto z listy: {current_node.robot}")
                    self.nodes[current_idx] = None
                    self.free_indices.append(current_idx)
                    self.size -= 1
                    return True
                
                prev_idx = current_idx
                current_idx = current_node.next
        
        print(f"Robot o kluczu {key} nie został znaleziony.")
        return False
    
    def search(self, key: str) -> Optional[Robot]:
        for head_idx in self.head_indices.values():
            current_idx = head_idx
            
            while current_idx is not None:
                current_node = self.nodes[current_idx]
                if current_node.key == key:
                    print(f"Znaleziono: {current_node.robot}")
                    return current_node.robot
                current_idx = current_node.next
        
        print(f"Robot o kluczu {key} nie został znaleziony.")
        return None
    
    def display(self):
        print("Zawartość listy:")
        for typ, head_idx in self.head_indices.items():
            print(f"Typ {typ}:")
            current_idx = head_idx
            while current_idx is not None:
                current_node = self.nodes[current_idx]
                print(f"  - {current_node.robot} (klucz: {current_node.key})")
                current_idx = current_node.next


class UnionFind:
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) 
        return self.parent[x]
    
    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


def generate_random_robot() -> Robot:
    typ = random.choice(Robot.TYPY)
    cena = round(random.uniform(0, 10000), 2)
    zasieg = random.randint(0, 100)
    kamera = random.randint(0, 1)
    return Robot(typ, cena, zasieg, kamera)


def generate_robot_list(n: int) -> List[Robot]:
    return [generate_random_robot() for _ in range(n)]


def display_robot_table(robots: List[Robot]):
    print("\n" + "="*80)
    print(f"{'Lp.':<4} {'Typ':<6} {'Cena (PLN)':<12} {'Zasięg (km)':<12} {'Kamera':<8}")
    print("="*80)
    
    for i, robot in enumerate(robots, 1):
        kamera_str = "TAK" if robot.kamera else "NIE"
        print(f"{i:<4} {robot.typ:<6} {robot.cena:<12.2f} {robot.zasieg:<12} {kamera_str:<8}")
    
    print("="*80)


def save_robots_to_file(robots: List[Robot], filename: str = "robots.json"):
    data = [robot.to_dict() for robot in robots]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Zapisano {len(robots)} robotów do pliku {filename}")


def load_robots_from_file(filename: str = "robots.json") -> List[Robot]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        robots = [Robot.from_dict(robot_data) for robot_data in data]
        print(f"Wczytano {len(robots)} robotów z pliku {filename}")
        return robots
    except FileNotFoundError:
        print(f"Plik {filename} nie istnieje.")
        return []


def get_robot_from_user() -> Robot:
    print("Podaj parametry robota:")
    
    while True:
        typ = input(f"Typ {Robot.TYPY}: ").upper()
        if typ in Robot.TYPY:
            break
        print("Nieprawidłowy typ!")
    
    while True:
        try:
            cena = float(input("Cena (0-10000 PLN): "))
            if 0 <= cena <= 10000:
                break
            print("Cena musi być z przedziału 0-10000!")
        except ValueError:
            print("Podaj liczbę!")
    
    while True:
        try:
            zasieg = int(input("Zasięg (0-100 km): "))
            if 0 <= zasieg <= 100:
                break
            print("Zasięg musi być z przedziału 0-100!")
        except ValueError:
            print("Podaj liczbę całkowitą!")
    
    while True:
        try:
            kamera = int(input("Kamera (0-nie ma, 1-ma): "))
            if kamera in [0, 1]:
                break
            print("Podaj 0 lub 1!")
        except ValueError:
            print("Podaj 0 lub 1!")
    
    return Robot(typ, cena, zasieg, kamera)


def create_sample_graph() -> Dict[int, List[int]]:
    print("Tworzenie przykładowego grafu...")
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
        3: [4],
        4: [3, 5],
        5: [4],
        6: [7],
        7: [6],
        8: []
    }
    return graph


def find_connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    if not graph:
        return []
    
    vertices = list(graph.keys())
    n = len(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    uf = UnionFind(n)
    
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor in vertex_to_index:
                uf.union(vertex_to_index[vertex], vertex_to_index[neighbor])
    
    components = {}
    for vertex in vertices:
        root = uf.find(vertex_to_index[vertex])
        if root not in components:
            components[root] = []
        components[root].append(vertex)
    
    return list(components.values())


def zadanie1():
    print("ZADANIE 1 - PRZYGOTOWANIE DANYCH")
    
    while True:
        try:
            n = int(input("Podaj liczbę robotów do wygenerowania: "))
            if n > 0:
                break
            print("Liczba musi być większa od 0!")
        except ValueError:
            print("Podaj liczbę całkowitą!")
    
    robots = generate_robot_list(n)
    
    display_robot_table(robots)
    
    save_robots_to_file(robots)
    
    return robots


def zadanie2():
    print("ZADANIE 2 - STOS ROBOTÓW")
    
    stack = RobotStack()
    
    while True:
        print("\nOpcje:")
        print("1. Dodaj robota do stosu")
        print("2. Usuń robota ze stosu")
        print("3. Wyczyść stos")
        print("4. Pokaż rozmiar stosu")
        print("5. Zakończ")
        
        choice = input("Wybierz opcję (1-5): ")
        
        if choice == '1':
            robot = get_robot_from_user()
            stack.push(robot)
        elif choice == '2':
            stack.pop()
        elif choice == '3':
            stack.clear()
        elif choice == '4':
            print(f"Rozmiar stosu: {stack.size()}")
        elif choice == '5':
            break
        else:
            print("Nieprawidłowa opcja!")


def zadanie3():
    print("ZADANIE 3 - KOLEJKA ROBOTÓW")
    queue = RobotQueue()
    
    while True:
        print("\nOpcje:")
        print("1. Dodaj robota do kolejki")
        print("2. Usuń robota z kolejki")
        print("3. Wyczyść kolejkę")
        print("4. Pokaż rozmiar kolejki")
        print("5. Zakończ")
        
        choice = input("Wybierz opcję (1-5): ")
        
        if choice == '1':
            robot = get_robot_from_user()
            queue.enqueue(robot)
        elif choice == '2':
            queue.dequeue()
        elif choice == '3':
            queue.clear()
        elif choice == '4':
            print(f"Rozmiar kolejki: {queue.size()}")
        elif choice == '5':
            break
        else:
            print("Nieprawidłowa opcja!")


def zadanie4():
    print("ZADANIE 4 - LISTA Z DOWIĄZANIAMI")
    
    linked_list = RobotLinkedList()
    
    while True:
        print("\nOpcje:")
        print("1. Dodaj robota do listy")
        print("2. Usuń robota z listy")
        print("3. Wyszukaj robota")
        print("4. Wyświetl listę")
        print("5. Zakończ")
        
        choice = input("Wybierz opcję (1-5): ")
        
        if choice == '1':
            robot = get_robot_from_user()
            linked_list.add(robot)
        elif choice == '2':
            key = input("Podaj klucz robota do usunięcia: ")
            linked_list.remove(key)
        elif choice == '3':
            key = input("Podaj klucz robota do wyszukania: ")
            linked_list.search(key)
        elif choice == '4':
            linked_list.display()
        elif choice == '5':
            break
        else:
            print("Nieprawidłowa opcja!")


def zadanie5():
    print("ZADANIE 5 - ZBIORY ROZŁĄCZNE I SPÓJNE SKŁADOWE")
    
    graph = create_sample_graph()
    
    print("Graf (lista sąsiedztwa):")
    for vertex, neighbors in graph.items():
        print(f"Wierzchołek {vertex}: {neighbors}")
    
    components = find_connected_components(graph)
    
    print(f"\nZnaleziono {len(components)} spójnych składowych:")
    for i, component in enumerate(components, 1):
        print(f"Składowa {i}: {sorted(component)}")
    
    vertices = list(graph.keys())
    if len(vertices) >= 2:
        uf = UnionFind(len(vertices))
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                if neighbor in vertex_to_index:
                    uf.union(vertex_to_index[vertex], vertex_to_index[neighbor])
        
        print("\nTesty połączenia wierzchołków:")
        test_pairs = [(0, 1), (0, 3), (2, 4), (6, 7), (3, 8)]
        for v1, v2 in test_pairs:
            if v1 in vertex_to_index and v2 in vertex_to_index:
                connected = uf.connected(vertex_to_index[v1], vertex_to_index[v2])
                print(f"Wierzchołki {v1} i {v2}: {'POŁĄCZONE' if connected else 'ROZŁĄCZONE'}")

if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()


