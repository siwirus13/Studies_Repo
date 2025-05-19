import math
import random
import time
import matplotlib.pyplot as plt
from typing import List
import os


# Zadanie 1: Ciągi liczbowe
def seq1_recursive(n):
    if n == 0:
        return 1
    return 3 * n + seq1_recursive(n - 1)

def seq1_formula(n):
    return 1 + 3 * n * (n + 1) // 2

def seq2_recursive(n):
    if n == 0 or n == -1:
        return 0
    return n + seq2_recursive(n - 2)

def seq2_formula(n):
    if n % 2 == 0:
        k = n // 2
        return k * (k + 1)
    else:
        k = n // 2
        return (k + 1) ** 2

def fib_recursive(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_formula(n):
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return round((phi**n - psi**n) / math.sqrt(5))

def zadanie1():
    N = int(input("Zadanie 1 - Podaj N: "))
    print(f"{'n':>3} | {'Seq1 Rec':>8} {'Seq1 For':>8} | {'Seq2 Rec':>8} {'Seq2 For':>8} | {'Fib Rec':>8} {'Fib For':>8}")
    print("-" * 65)
    for n in range(N + 1):
        s1r, s1f = seq1_recursive(n), seq1_formula(n)
        s2r, s2f = seq2_recursive(n), seq2_formula(n)
        fr, ff = fib_recursive(n), fib_formula(n)
        print(f"{n:>3} | {s1r:>8} {s1f:>8} | {s2r:>8} {s2f:>8} | {fr:>8} {ff:>8}")

# Zadanie 2: Operacje na listach
def max_recursive(lst: List[int]) -> int:
    if len(lst) == 1:
        return lst[0]
    mid = len(lst) // 2
    left = max_recursive(lst[:mid])
    right = max_recursive(lst[mid:])
    return max(left, right)

def second_max_recursive(lst: List[int]) -> int:
    unique = list(set(lst))
    if len(unique) < 2:
        return unique[0]
    unique.remove(max_recursive(unique))
    return max_recursive(unique)

def average_recursive(lst: List[int]) -> float:
    def helper(sub):
        if len(sub) == 1:
            return sub[0], 1
        mid = len(sub) // 2
        sum1, count1 = helper(sub[:mid])
        sum2, count2 = helper(sub[mid:])
        return sum1 + sum2, count1 + count2
    total, count = helper(lst)
    return total / count

def zadanie2():
    lst = list(map(int, input("Zadanie 2 — Podaj listę liczb (oddzielone spacjami): ").split()))
    print("Największy:", max_recursive(lst))
    print("Drugi największy:", second_max_recursive(lst))
    print("Średnia:", average_recursive(lst))

# Zadanie 3: Merge sort
def merge_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def zadanie3():
    lst = list(map(int, input("Zadanie 3 — Podaj listę liczb (oddzielone spacjami): ").split()))
    print("Posortowana:", merge_sort(lst))


# Zadanie 4


def load_graph(filepath):
    """
    Wczytuje graf z pliku. Plik powinien zawierać po jednej krawędzi na linię, np.:
    A B
    B C
    C D
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plik nie istnieje: {filepath}")

    graph = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # Pomijamy błędne linie
            a, b = parts[0].strip(), parts[1].strip()
            if a not in graph:
                graph[a] = []
            if b not in graph:
                graph[b] = []
            graph[a].append(b)
            graph[b].append(a)  # Graf nieskierowany
    return graph




def shortest_path_recursive(graph, start, end, visited=None):
    if visited is None:
        visited = set()
    if start == end:
        return [start]
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            path = shortest_path_recursive(graph, neighbor, end, visited.copy())
            if path:
                return [start] + path
    return None


def zadanie4():
    filepath = input("Zadanie 4 — Podaj ścieżkę do pliku z grafem: ").strip()
    try:
        graph = load_graph(filepath)
    except FileNotFoundError as e:
        print(e)
        return

    start = input("Podaj wierzchołek startowy: ").strip()
    end = input("Podaj wierzchołek końcowy: ").strip()

    if start not in graph or end not in graph:
        print("Podane wierzchołki nie istnieją w grafie.")
        return

    path = shortest_path_recursive(graph, start, end)
    if path:
        print("Najkrótsza ścieżka:", " -> ".join(path))
        print("Długość:", len(path) - 1)
    else:
        print("Brak ścieżki.")





 #Zadanie 5: Wykresy czasów działania
def measure_time(func, arg):
    start = time.time()
    func(arg)
    return time.time() - start

def zadanie5():
    sizes = [2**i for i in range(4, 13)]
    times_max = []
    times_avg = []
    times_sort = []

    for size in sizes:
        data = [random.randint(0, 1000) for _ in range(size)]
        times_max.append(measure_time(max_recursive, data))
        times_avg.append(measure_time(average_recursive, data))
        times_sort.append(measure_time(merge_sort, data))

    plt.plot(sizes, times_max, label="Max (rekurencyjnie)")
    plt.plot(sizes, times_avg, label="Average (rekurencyjnie)")
    plt.plot(sizes, times_sort, label="Merge Sort")
    plt.xlabel("Rozmiar listy")
    plt.ylabel("Czas działania [s]")
    plt.title("Zadanie 5 — Czas działania algorytmów")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
