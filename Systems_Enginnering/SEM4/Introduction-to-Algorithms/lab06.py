import math
import random
import time
import matplotlib.pyplot as plt

def zadanie1(n):
    def rozklad(n, dzielnik=2):
        if n == 1:
            return []
        if n % dzielnik == 0:
            return [dzielnik] + rozklad(n // dzielnik, dzielnik)
        return rozklad(n, dzielnik + 1)
    return rozklad(n)

def zadanie2(p):
    x = [1] * (p + 1)
    x[0:2] = [0, 0]
    for n in range(2, int(math.sqrt(p)) + 1):
        if x[n] == 1:
            for j in range(2 * n, p + 1, n):
                x[j] = 0
    return [i for i, val in enumerate(x) if val == 1]

def RNWD(a, b):
    def czynniki(n):
        return zadanie1(n)
    from collections import Counter
    ca = Counter(czynniki(a))
    cb = Counter(czynniki(b))
    wspolne = ca & cb
    nwd = 1
    for k in wspolne:
        nwd *= k ** wspolne[k]
    return nwd

def ENWD(a, b):
    while b:
        a, b = b, a % b
    return a

def zadanie3(n, m):
    rnwd_times = []
    enwd_times = []
    q_values = range(1, m + 1)
    for q in q_values:
        start = time.perf_counter()
        RNWD(n, q)
        rnwd_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        ENWD(n, q)
        enwd_times.append(time.perf_counter() - start)

    plt.plot(q_values, rnwd_times, label="RNWD (faktoryzacja)")
    plt.plot(q_values, enwd_times, label="ENWD (Euklides)")
    plt.xlabel("q")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.title("Porównanie czasu działania RNWD i ENWD")
    plt.grid(True)
    plt.show()

def szybkie_potegowanie(a, d, n):
    wynik = 1
    a = a % n
    while d > 0:
        if d % 2 == 1:
            wynik = (wynik * a) % n
        d = d // 2
        a = (a * a) % n
    return wynik

def test_fermata(p, k=5):
    if p <= 3:
        return p == 2 or p == 3
    for _ in range(k):
        a = random.randint(2, p - 2)
        if szybkie_potegowanie(a, p - 1, p) != 1:
            return False
    return True

def test_millera_rabina(p, k=5):
    if p == 2 or p == 3:
        return True
    if p % 2 == 0 or p < 2:
        return False
    d = p - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = random.randint(2, p - 2)
        x = szybkie_potegowanie(a, d, p)
        if x == 1 or x == p - 1:
            continue
        for _ in range(r - 1):
            x = szybkie_potegowanie(x, 2, p)
            if x == p - 1:
                break
        else:
            return False
    return True

def zadanie4(p):
    return {
        "fermata": test_fermata(p),
        "miller_rabin": test_millera_rabina(p)
    }

# Zadanie 5 - RSA

def egcd(a, b):
    if a == 0:
        return b, 0, 1
    g, y, x = egcd(b % a, a)
    return g, x - (b // a) * y, y

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Brak odwrotności modulo')
    return x % m

def generuj_klucze_rsa(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    d = modinv(e, phi)
    return (e, n), (d, n)

def szyfruj_rsa(tekst, klucz):
    e, n = klucz
    return [pow(ord(znak), e, n) for znak in tekst]

def odszyfruj_rsa(szyfrogram, klucz):
    d, n = klucz
    return ''.join([chr(pow(c, d, n)) for c in szyfrogram])

def zadanie5():
    p = int(input("Podaj pierwszą liczbę pierwszą (p): "))
    q = int(input("Podaj drugą liczbę pierwszą (q): "))
    tekst = input("Podaj tekst do zaszyfrowania: ")
    publiczny, prywatny = generuj_klucze_rsa(p, q)
    zaszyfrowany = szyfruj_rsa(tekst, publiczny)
    odszyfrowany = odszyfruj_rsa(zaszyfrowany, prywatny)
    print("Klucz publiczny:", publiczny)
    print("Klucz prywatny:", prywatny)
    print("Zaszyfrowany tekst:", zaszyfrowany)
    print("Odszyfrowany tekst:", odszyfrowany)

if __name__ == "__main__":
    n1 = int(input("Podaj liczbę do rozkładu na czynniki: "))
    print("Zadanie 1: Rozkład na czynniki", zadanie1(n1))

    p = int(input("Podaj liczbę do sita Eratostenesa: "))
    print("Zadanie 2: Sito Eratostenesa", zadanie2(p))

    a = int(input("Podaj pierwszą liczbę do obliczenia NWD: "))
    b = int(input("Podaj drugą liczbę do obliczenia NWD: "))
    print("Zadanie 3.1: RNWD, ENWD", RNWD(a, b), ENWD(a, b))

    test_p = int(input("Podaj liczbę do testów pierwszości: "))
    print("Zadanie 4: Testy pierwszości", zadanie4(test_p))

    n = int(input("Podaj liczbę n do testu wydajności: "))
    m = int(input("Podaj maksymalne q do testu wydajności: "))
    zadanie3(n, m)

    zadanie5()
