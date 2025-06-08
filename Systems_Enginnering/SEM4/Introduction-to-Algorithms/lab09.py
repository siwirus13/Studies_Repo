import random
import json
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
import statistics


class Robot:
    """Klasa reprezentująca pojedynczy robot mobilny."""
    
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
        """Konwersja do słownika dla zapisu do pliku."""
        return {
            'typ': self.typ,
            'cena': self.cena,
            'zasieg': self.zasieg,
            'kamera': self.kamera
        }
    
    @classmethod
    def from_dict(cls, data):
        """Tworzenie robota ze słownika."""
        return cls(data['typ'], data['cena'], data['zasieg'], data['kamera'])
    
    def get_parameter(self, param_name: str):
        """Zwraca wartość parametru o podanej nazwie."""
        if param_name.upper() == 'TYP':
            return self.typ
        elif param_name.upper() == 'CENA':
            return self.cena
        elif param_name.upper() == 'ZASIĘG' or param_name.upper() == 'ZASIEG':
            return self.zasieg
        elif param_name.upper() == 'KAMERA':
            return self.kamera
        else:
            raise ValueError(f"Nieznany parametr: {param_name}")


def generate_random_robot() -> Robot:
    """Generuje losowego robota."""
    typ = random.choice(Robot.TYPY)
    cena = round(random.uniform(0, 10000), 2)
    zasieg = random.randint(0, 100)
    kamera = random.randint(0, 1)
    return Robot(typ, cena, zasieg, kamera)


def generate_robot_list(n: int) -> List[Robot]:
    """Generuje listę N robotów o losowych parametrach."""
    return [generate_random_robot() for _ in range(n)]


def save_robots_to_file(robots: List[Robot], filename: str = "robots.json"):
    """Zapisuje listę robotów do pliku."""
    data = [robot.to_dict() for robot in robots]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Zapisano {len(robots)} robotów do pliku {filename}")


def load_robots_from_file(filename: str = "robots.json") -> List[Robot]:
    """Wczytuje listę robotów z pliku."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        robots = [Robot.from_dict(robot_data) for robot_data in data]
        print(f"Wczytano {len(robots)} robotów z pliku {filename}")
        return robots
    except FileNotFoundError:
        print(f"Plik {filename} nie istnieje. Generuję nową listę robotów.")
        robots = generate_robot_list(100)
        save_robots_to_file(robots, filename)
        return robots


def linear_search(robots: List[Robot], search_criteria: List[Union[str, float, int, List, None]]) -> Optional[Robot]:
    """
    Wyszukiwanie liniowe robota według kryteriów.
    
    Args:
        robots: Lista robotów
        search_criteria: [typ, cena, zasięg, kamera] - każdy element może być:
                        - konkretną wartością
                        - listą dopuszczalnych wartości
                        - None (dowolna wartość)
    
    Returns:
        Pierwszy znaleziony robot lub None
    """
    typ_criteria, cena_criteria, zasieg_criteria, kamera_criteria = search_criteria
    
    for robot in robots:
        if typ_criteria is not None:
            if isinstance(typ_criteria, list):
                if robot.typ not in typ_criteria:
                    continue
            else:
                if robot.typ != typ_criteria:
                    continue
        
        if cena_criteria is not None:
            if isinstance(cena_criteria, list):
                if robot.cena not in cena_criteria:
                    continue
            else:
                if robot.cena != cena_criteria:
                    continue
        
        if zasieg_criteria is not None:
            if isinstance(zasieg_criteria, list):
                if robot.zasieg not in zasieg_criteria:
                    continue
            else:
                if robot.zasieg != zasieg_criteria:
                    continue
        
        if kamera_criteria is not None:
            if isinstance(kamera_criteria, list):
                if robot.kamera not in kamera_criteria:
                    continue
            else:
                if robot.kamera != kamera_criteria:
                    continue
        
        return robot
    
    return None


def binary_search(sorted_robots: List[Robot], param_name: str, target_values: List[Union[str, float, int]]) -> Optional[Robot]:
    """
    Wyszukiwanie binarne robota według parametru.
    
    Args:
        sorted_robots: Lista robotów posortowana według param_name
        param_name: Nazwa parametru do wyszukiwania
        target_values: Lista poszukiwanych wartości
    
    Returns:
        Pierwszy znaleziony robot lub None
    """
    for target_value in target_values:
        left, right = 0, len(sorted_robots) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_value = sorted_robots[mid].get_parameter(param_name)
            
            if mid_value == target_value:
                return sorted_robots[mid]
            elif mid_value < target_value:
                left = mid + 1
            else:
                right = mid - 1
    
    return None


class ChainHashTable:
    """Tablica haszująca z metodą łańcuchową."""
    
    def __init__(self, size: int):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key: str) -> int:
        """Funkcja haszująca."""
        hash_value = 0
        for char in str(key):
            hash_value = (hash_value * 31 + ord(char)) % self.size
        return hash_value
    
    def _generate_key(self, robot: Robot) -> str:
        """Generuje klucz dla robota."""
        return f"{robot.typ}_{robot.cena:.2f}_{robot.zasieg}_{robot.kamera}"
    
    def insert(self, robot: Robot):
        """Wstawia robota do tablicy."""
        key = self._generate_key(robot)
        index = self._hash(key)
        
        for existing_key, existing_robot in self.table[index]:
            if existing_key == key:
                return  
        
        self.table[index].append((key, robot))
    
    def search(self, param_name: str, param_value: Union[str, float, int]) -> Optional[Robot]:
        """Wyszukuje robota według parametru."""
        # Przeszukaj wszystkie łańcuchy
        for chain in self.table:
            for key, robot in chain:
                if robot.get_parameter(param_name) == param_value:
                    return robot
        return None


class OpenAddressingHashTable:
    """Tablica haszująca z adresowaniem otwartym kwadratowym."""
    
    def __init__(self, size: int):
        self.size = size
        self.table = [None] * size
        self.deleted = [False] * size
    
    def _hash1(self, key: str) -> int:
        """Pierwsza funkcja haszująca."""
        hash_value = 0
        for char in str(key):
            hash_value = (hash_value * 31 + ord(char)) % self.size
        return hash_value
    
    def _hash2(self, key: str) -> int:
        """Druga funkcja haszująca dla podwójnego haszowania."""
        hash_value = 0
        for char in str(key):
            hash_value = (hash_value * 37 + ord(char)) % (self.size - 1)
        return hash_value + 1
    
    def _generate_key(self, robot: Robot) -> str:
        """Generuje klucz dla robota."""
        return f"{robot.typ}_{robot.cena:.2f}_{robot.zasieg}_{robot.kamera}"
    
    def insert(self, robot: Robot):
        """Wstawia robota do tablicy."""
        key = self._generate_key(robot)
        index = self._hash1(key)
        
        i = 0
        while i < self.size:
            pos = (index + i * i) % self.size
            
            if self.table[pos] is None or self.deleted[pos]:
                self.table[pos] = (key, robot)
                self.deleted[pos] = False
                return True
            
            # Sprawdź czy robot już istnieje
            if self.table[pos][0] == key:
                return False
            
            i += 1
        
        return False  # Tablica pełna
    
    def search(self, param_name: str, param_value: Union[str, float, int]) -> Optional[Robot]:
        """Wyszukuje robota według parametru."""
        for i in range(self.size):
            if self.table[i] is not None and not self.deleted[i]:
                key, robot = self.table[i]
                if robot.get_parameter(param_name) == param_value:
                    return robot
        return None


def zadanie1():
    robots = load_robots_from_file()
    
    test_cases = [
        (["AGV", None, [5, 6, 7, 8, 9, 10], 1], "Robot AGV z kamerą i zasięgiem 5-10"),
        (["AFV", None, None, 0], "Robot AFV bez kamery"),
        ([None, None, None, 1], "Dowolny robot z kamerą"),
        (["AUV", [5000, 6000, 7000], None, None], "Robot AUV o cenie 5000, 6000 lub 7000")
    ]
    
    for criteria, description in test_cases:
        print(f"\nSzukam: {description}")
        print(f"Kryteria: {criteria}")
        
        start_time = time.time()
        result = linear_search(robots, criteria)
        end_time = time.time()
        
        if result:
            print(f"Znaleziono: {result}")
        else:
            print("Brak")
        print(f"Czas wyszukiwania: {(end_time - start_time)*1000:.4f} ms")


def zadanie2():
    robots = load_robots_from_file()
    
    parameters = ['TYP', 'CENA', 'ZASIĘG', 'KAMERA']
    
    for param in parameters:
        print(f"\n--- Wyszukiwanie według parametru: {param} ---")
        
        sorted_robots = sorted(robots, key=lambda r: r.get_parameter(param))
        
        if param == 'TYP':
            search_values = ['AGV', 'AFV']
        elif param == 'CENA':
            search_values = [1000, 5000, 9000]
        elif param == 'ZASIĘG':
            search_values = [10, 50, 90]
        else:  # KAMERA
            search_values = [0, 1]
        
        start_time = time.time()
        result = binary_search(sorted_robots, param, search_values)
        end_time = time.time()
        
        if result:
            print(f"Znaleziono: {result}")
        else:
            print("Brak")
        print(f"Czas wyszukiwania: {(end_time - start_time)*1000:.4f} ms")


def zadanie3():
    robots = load_robots_from_file()
    
    alpha = 0.75
    table_size = int(len(robots) / alpha)
    
    print(f"Liczba robotów: {len(robots)}")
    print(f"Współczynnik wypełnienia α: {alpha}")
    print(f"Rozmiar tablicy: {table_size}")
    
    hash_table = ChainHashTable(table_size)
    
    for robot in robots:
        hash_table.insert(robot)
    
    test_searches = [
        ('TYP', 'AGV'),
        ('CENA', 5000.0),
        ('ZASIĘG', 50),
        ('KAMERA', 1)
    ]
    
    for param, value in test_searches:
        print(f"\nSzukam robota z {param} = {value}")
        
        start_time = time.time()
        result = hash_table.search(param, value)
        end_time = time.time()
        
        if result:
            print(f"Znaleziono: {result}")
        else:
            print("Brak")
        print(f"Czas wyszukiwania: {(end_time - start_time)*1000:.4f} ms")


def zadanie4():
    robots = load_robots_from_file()
    
    alpha = 0.7
    table_size = int(len(robots) / alpha)
    
    print(f"Liczba robotów: {len(robots)}")
    print(f"Współczynnik wypełnienia α: {alpha}")
    print(f"Rozmiar tablicy: {table_size}")
    
    hash_table = OpenAddressingHashTable(table_size)
    
    inserted = 0
    for robot in robots:
        if hash_table.insert(robot):
            inserted += 1
    
    print(f"Wstawiono robotów: {inserted}")
    
    test_searches = [
        ('TYP', 'AGV'),
        ('CENA', 5000.0),
        ('ZASIĘG', 50),
        ('KAMERA', 1)
    ]
    
    for param, value in test_searches:
        print(f"\nSzukam robota z {param} = {value}")
        
        start_time = time.time()
        result = hash_table.search(param, value)
        end_time = time.time()
        
        if result:
            print(f"Znaleziono: {result}")
        else:
            print("Brak")
        print(f"Czas wyszukiwania: {(end_time - start_time)*1000:.4f} ms")


def make_unique_prices(robots: List[Robot]) -> List[Robot]:
    """Sprawia, że ceny robotów są unikatowe."""
    used_prices = set()
    unique_robots = []
    
    for robot in robots:
        original_price = robot.cena
        price = original_price
        increment = 0.01
        
        # Znajdź pierwszą dostępną cenę
        while price in used_prices:
            price = original_price + increment
            increment += 0.01
        
        used_prices.add(price)
        robot.cena = price
        unique_robots.append(robot)
    
    return unique_robots


def measure_search_performance(robots: List[Robot], alpha_values: List[float]) -> Dict[str, Dict[float, List[float]]]:
    """Mierzy wydajność algorytmów wyszukiwania dla różnych wartości α."""
    results = {
        'chain': defaultdict(list),
        'open_addressing': defaultdict(list)
    }
    
    sorted_robots = sorted(robots, key=lambda r: r.cena)
    
    for alpha in alpha_values:
        print(f"\nTestowanie dla α = {alpha}")
        table_size = max(int(len(robots) / alpha), len(robots) + 1)
        
        chain_table = ChainHashTable(table_size)
        for robot in sorted_robots:
            chain_table.insert(robot)
        
        open_table = OpenAddressingHashTable(table_size)
        for robot in sorted_robots:
            open_table.insert(robot)
        
        chain_times = []
        open_times = []
        
        for robot in sorted_robots[:min(50, len(sorted_robots))]:  # Ogranicz do 50 testów
            start_time = time.perf_counter()
            chain_table.search('CENA', robot.cena)
            end_time = time.perf_counter()
            chain_times.append((end_time - start_time) * 1000000)  # w mikrosekundach
            
            start_time = time.perf_counter()
            open_table.search('CENA', robot.cena)
            end_time = time.perf_counter()
            open_times.append((end_time - start_time) * 1000000)  # w mikrosekundach
        
        results['chain'][alpha] = chain_times
        results['open_addressing'][alpha] = open_times
    
    return results


def zadanie5():
    robots = load_robots_from_file()
    
    robots = make_unique_prices(robots)
    print(f"Przygotowano {len(robots)} robotów z unikatowymi cenami")
    
    robots.sort(key=lambda r: r.cena)
    
    alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Mierzenie wydajności algorytmów...")
    results = measure_search_performance(robots, alpha_values)
    
    chain_means = []
    open_means = []
    
    for alpha in alpha_values:
        chain_mean = statistics.mean(results['chain'][alpha])
        open_mean = statistics.mean(results['open_addressing'][alpha])
        
        chain_means.append(chain_mean)
        open_means.append(open_mean)
        
        print(f"α = {alpha}: Łańcuchowa = {chain_mean:.2f}μs, Otwarte = {open_mean:.2f}μs")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(alpha_values, chain_means, 'b-o', label='Metoda łańcuchowa', linewidth=2, markersize=8)
    plt.plot(alpha_values, open_means, 'r-s', label='Adresowanie otwarte', linewidth=2, markersize=8)
    plt.xlabel('Współczynnik wypełnienia α')
    plt.ylabel('Średni czas wyszukiwania (μs)')
    plt.title('Porównanie wydajności algorytmów haszowania')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    x = range(len(alpha_values))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], chain_means, width, label='Metoda łańcuchowa', alpha=0.8)
    plt.bar([i + width/2 for i in x], open_means, width, label='Adresowanie otwarte', alpha=0.8)
    
    plt.xlabel('Współczynnik wypełnienia α')
    plt.ylabel('Średni czas wyszukiwania (μs)')
    plt.title('Porównanie wydajności - wykres słupkowy')
    plt.xticks(x, [f'{a}' for a in alpha_values])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    print("Zadanie 1:")
    zadanie1()
    print("Zadanie 2:")
    zadanie2()
    print("Zadanie 3:")
    zadanie3()
    print("Zadanie 4:")
    zadanie4()
    print("Zadanie 5:")
    zadanie5()

