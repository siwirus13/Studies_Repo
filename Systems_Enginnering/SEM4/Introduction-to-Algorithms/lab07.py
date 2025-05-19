import numpy as np
import matplotlib.pyplot as plt


def zadanie1(p1, p2):
    n, m = len(p1), len(p2)
    result = [0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += p1[i] * p2[j]
    return result


def zadanie2(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = zadanie2(signal[0::2])
    odd = zadanie2(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]


def zadanie3(p1, p2):
    n = 1
    while n < len(p1) + len(p2) - 1:
        n *= 2
    p1 += [0] * (n - len(p1))
    p2 += [0] * (n - len(p2))
    fft1 = zadanie2(p1)
    fft2 = zadanie2(p2)
    fft_product = [a * b for a, b in zip(fft1, fft2)]
    result = zadanie2([x.conjugate() for x in fft_product])
    result = [x.conjugate() / n for x in result]
    return [round(r.real, 5) for r in result]


def zadanie4():
    import time
    import random

    sizes = [2**i for i in range(2, 12)]
    naive_times = []
    fft_times = []

    for size in sizes:
        a = [random.random() for _ in range(size)]
        b = [random.random() for _ in range(size)]

        start = time.time()
        zadanie1(a, b)
        naive_times.append(time.time() - start)

        start = time.time()
        zadanie3(a.copy(), b.copy())
        fft_times.append(time.time() - start)

    plt.plot(sizes, naive_times, label="Naive")
    plt.plot(sizes, fft_times, label="FFT-based")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Time [s]")
    plt.title("Comparison of Polynomial Multiplication Methods")
    plt.legend()
    plt.grid(True)
    plt.show()


def pad_to_power_of_two(signal):
    n = len(signal)
    power = 1
    while power < n:
        power *= 2
    return signal + [0] * (power - n)


def zadanie5(Ai, ai, Bj, bj):
    t = np.linspace(0, 1, 1024)
    signal = np.zeros_like(t)

    for A, a in zip(Ai, ai):
        signal += A * np.sin(2 * np.pi * a * t)

    for B, b in zip(Bj, bj):
        signal += B * np.cos(2 * np.pi * b * t)

    padded_signal = pad_to_power_of_two(signal.tolist())
    freq_signal = zadanie2(padded_signal)

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title("Sygnał oryginalny")
    plt.grid(True)
    plt.show()

    freqs = np.fft.fftfreq(len(padded_signal), d=t[1] - t[0])
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(freqs), np.abs(freq_signal))
    plt.title("Widmo amplitudowe przed filtracją")
    plt.grid(True)
    plt.show()

    to_remove = input("Podaj częstotliwości do usunięcia (oddzielone spacją): ")
    to_remove = list(map(float, to_remove.split()))

    for i, f in enumerate(freqs):
        if any(abs(f - r) < 1e-3 for r in to_remove):
            freq_signal[i] = 0

    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(freqs), np.abs(freq_signal))
    plt.title("Widmo amplitudowe po filtracji")
    plt.grid(True)
    plt.show()

    inv = zadanie2([x.conjugate() for x in freq_signal])
    inv = [x.conjugate().real / len(freq_signal) for x in inv]

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Przed filtracją")
    plt.plot(t, inv[:len(t)], label="Po filtracji")
    plt.legend()
    plt.title("Sygnał przed i po filtracji")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("=== Zadanie 1 ===")
    poly1 = list(map(float, input("Podaj współczynniki pierwszego wielomianu (np. 1 2 3): ").split()))
    poly2 = list(map(float, input("Podaj współczynniki drugiego wielomianu (np. 1 0 1): ").split()))
    print("Wynik mnożenia (naiwnie):")
    print(zadanie1(poly1, poly2))

    print("\n=== Zadanie 2 ===")
    signal = list(map(float, input("Podaj sygnał (np. 1 0 1 0): ").split()))
    signal = pad_to_power_of_two(signal)
    print("FFT:")
    print(zadanie2(signal))

    print("\n=== Zadanie 3 ===")
    poly1 = list(map(float, input("Podaj współczynniki pierwszego wielomianu: ").split()))
    poly2 = list(map(float, input("Podaj współczynniki drugiego wielomianu: ").split()))
    print("Wynik mnożenia (FFT):")
    print(zadanie3(poly1, poly2))

    print("\n=== Zadanie 4 ===")
    zadanie4()

    print("\n=== Zadanie 5 ===")
    Ai = list(map(float, input("Podaj współczynniki Ai (dla sinusów, np. 1 0.5): ").split()))
    ai = list(map(float, input("Podaj częstotliwości ai (dla sinusów, np. 5 15): ").split()))
    Bj = list(map(float, input("Podaj współczynniki Bj (dla cosinusów, np. 1 0.2): ").split()))
    bj = list(map(float, input("Podaj częstotliwości bj (dla cosinusów, np. 10 20): ").split()))

    min_len = min(len(Ai), len(ai), len(Bj), len(bj))
    Ai, ai, Bj, bj = Ai[:min_len], ai[:min_len], Bj[:min_len], bj[:min_len]
    zadanie5(Ai, ai, Bj, bj)
