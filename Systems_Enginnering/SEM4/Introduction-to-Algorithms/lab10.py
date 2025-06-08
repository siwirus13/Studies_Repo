
import json
import matplotlib.pyplot as plt
import time
from copy import deepcopy

def wczytaj_robota():
    with open("robots.json") as f:
        return json.load(f)

def zapisz_robota(roboty, nazwa="posortowane_roboty.json"):
    with open(nazwa, "w") as f:
        json.dump(roboty, f, indent=2)

def zadanie1():
    def heapify(arr, n, i, krok=False):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l]["cena"] > arr[largest]["cena"]:
            largest = l
        if r < n and arr[r]["cena"] > arr[largest]["cena"]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            if krok: print([r["cena"] for r in arr])
            heapify(arr, n, largest, krok)

    roboty = wczytaj_robota()
    n = len(roboty)
    for i in range(n // 2 - 1, -1, -1):
        heapify(roboty, n, i, krok=True)
    for i in range(n - 1, 0, -1):
        roboty[0], roboty[i] = roboty[i], roboty[0]
        print([r["cena"] for r in roboty])
        heapify(roboty, i, 0, krok=True)
    zapisz_robota(roboty)

def zadanie2():
    roboty = wczytaj_robota()
    def quicksort(arr, krok=False):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]["cena"]
        left = [x for x in arr[1:] if x["cena"] <= pivot]
        right = [x for x in arr[1:] if x["cena"] > pivot]
        result = quicksort(left, krok) + [arr[0]] + quicksort(right, krok)
        if krok: print([r["cena"] for r in result])
        return result
    roboty = quicksort(roboty, krok=True)
    zapisz_robota(roboty)

def zadanie3():
    roboty = wczytaj_robota()
    max_zasieg = max(r["zasieg"] for r in roboty)
    count = [[] for _ in range(max_zasieg + 1)]
    for r in roboty:
        count[r["zasieg"]].append(r)
    wynik = []
    for grupa in count:
        wynik.extend(grupa)
    zapisz_robota(wynik)

def zadanie4():
    M = int(input("Podaj liczbę wierszy: "))
    N = int(input("Podaj liczbę kolumn: "))
    macierz = []
    for _ in range(M):
        macierz.append(list(map(int, input().split())))
    macierz.sort()
    for wiersz in macierz:
        print(*wiersz)

def zadanie5():
    roboty1 = wczytaj_robota()
    roboty2 = deepcopy(roboty1)
    dane_heap, dane_quick = [], []

    def heapify(arr, n, i):
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and arr[l]["cena"] > arr[largest]["cena"]:
            largest = l
        if r < n and arr[r]["cena"] > arr[largest]["cena"]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            dane_heap.append([r["cena"] for r in arr])
            heapify(arr, n, largest)

    def heapsort(arr):
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            dane_heap.append([r["cena"] for r in arr])
            heapify(arr, i, 0)

    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]["cena"]
        left = [x for x in arr[1:] if x["cena"] <= pivot]
        right = [x for x in arr[1:] if x["cena"] > pivot]
        result = quicksort(left) + [arr[0]] + quicksort(right)
        dane_quick.append([r["cena"] for r in result])
        return result

    heapsort(roboty1)
    roboty2 = quicksort(roboty2)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Heap Sort")
    ax2.set_title("Quick Sort")

    for i in range(max(len(dane_heap), len(dane_quick))):
        ax1.clear()
        ax2.clear()
        if i < len(dane_heap):
            ax1.bar(range(len(dane_heap[i])), dane_heap[i], color='blue')
        if i < len(dane_quick):
            ax2.bar(range(len(dane_quick[i])), dane_quick[i], color='green')
        ax1.set_title("Heap Sort")
        ax2.set_title("Quick Sort")
        plt.pause(0.5)
    plt.show()

if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
