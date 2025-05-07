
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy import signal, fft
from pathlib import Path
import pandas as pd


def zadanie1():
    """Interaktywny wykres okna i jego widma z suwakiem długości okna oraz wyborem typu."""
    init_n = 64
    init_window = 'Hamming'
    n_range = (16, 512)

    def get_window(name: str, n: int) -> np.ndarray:
        return {
            'Hamming': signal.windows.hamming(n),
            'Hann': signal.windows.hann(n),
            'Blackman': signal.windows.blackman(n),
            'Dirichlet': signal.windows.boxcar(n)
        }[name]

    win = get_window(init_window, init_n)
    spectrum = 20 * np.log10(np.abs(np.fft.fft(win, 2048)) / np.max(np.abs(np.fft.fft(win, 2048))))
    freq = np.linspace(0, 1, len(spectrum))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.35, bottom=0.25)

    line_time, = ax1.plot(win, label='Okno czasowe')
    ax1.set_title("Okno czasowe")
    ax1.grid(True)

    line_freq, = ax2.plot(freq, spectrum)
    ax2.set_title("Widmo amplitudowe (dB)")
    ax2.set_ylim([-100, 5])
    ax2.grid(True)

    ax_n = plt.axes([0.35, 0.15, 0.55, 0.03])
    s_n = Slider(ax_n, 'N', n_range[0], n_range[1], valinit=init_n, valstep=1)

    radio_ax = plt.axes([0.05, 0.5, 0.2, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(radio_ax, ('Hamming', 'Hann', 'Blackman', 'Dirichlet'))
    current_window = [init_window]

    def update(val=None):
        n = int(s_n.val)
        wtype = current_window[0]
        win = get_window(wtype, n)
        spectrum = 20 * np.log10(np.abs(np.fft.fft(win, 2048)) / np.max(np.abs(np.fft.fft(win, 2048))))
        line_time.set_ydata(win)
        line_time.set_xdata(np.arange(len(win)))
        line_freq.set_ydata(spectrum)
        line_freq.set_xdata(np.linspace(0, 1, len(spectrum)))
        ax1.relim()
        ax1.autoscale_view()
        fig.canvas.draw_idle()

    s_n.on_changed(update)

    def change_window(label: str):
        current_window[0] = label
        update()

    radio.on_clicked(change_window)

    plt.show()


def zadanie2():
    """Widma sygnału sinusoidalnego z różnymi oknami (interaktywnie)"""
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    init_f1 = 50.0
    init_f2 = 75.0
    init_f3 = 100.0
    init_window = 'Hamming'

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.35, bottom=0.3)

    def get_window(name):
        return {
            'Hamming': signal.windows.hamming(len(t)),
            'Hann': signal.windows.hann(len(t)),
            'Blackman': signal.windows.blackman(len(t)),
            'Dirichlet': signal.windows.boxcar(len(t))
        }[name]

    def compute_and_plot(f1, f2, f3, wname):
        sig = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)
        win = get_window(wname)
        windowed = sig * win
        spectrum = np.abs(np.fft.fft(windowed))[:len(t) // 2]
        freqs = np.fft.fftfreq(len(t), 1 / fs)[:len(t) // 2]
        return freqs, 20 * np.log10(spectrum / np.max(spectrum))

    f, amp = compute_and_plot(init_f1, init_f2, init_f3, init_window)
    line, = ax.plot(f, amp)
    ax.set_ylim([-100, 5])
    ax.set_title('Widmo sygnału sinusoidalnego')
    ax.grid(True)

    ax_f1 = plt.axes([0.35, 0.2, 0.55, 0.03])
    ax_f2 = plt.axes([0.35, 0.15, 0.55, 0.03])
    ax_f3 = plt.axes([0.35, 0.1, 0.55, 0.03])

    s_f1 = Slider(ax_f1, 'f1 [Hz]', 1, 500, valinit=init_f1)
    s_f2 = Slider(ax_f2, 'f2 [Hz]', 1, 500, valinit=init_f2)
    s_f3 = Slider(ax_f3, 'f3 [Hz]', 1, 500, valinit=init_f3)

    radio_ax = plt.axes([0.05, 0.5, 0.2, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(radio_ax, ('Hamming', 'Hann', 'Blackman', 'Dirichlet'))
    current_window = [init_window]

    def update(val=None):
        f1, f2, f3 = s_f1.val, s_f2.val, s_f3.val
        freqs, amps = compute_and_plot(f1, f2, f3, current_window[0])
        line.set_xdata(freqs)
        line.set_ydata(amps)
        fig.canvas.draw_idle()

    def change_window(label):
        current_window[0] = label
        update()

    s_f1.on_changed(update)
    s_f2.on_changed(update)
    s_f3.on_changed(update)
    radio.on_clicked(change_window)

    plt.show()


def zadanie3():
    """FFT sygnału sinusoidalnego z interaktywną zmianą częstotliwości"""
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    init_f = 60.0

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    sig = np.sin(2 * np.pi * init_f * t)
    spectrum = np.abs(fft.fft(sig))[:len(t) // 2]
    freqs = np.fft.fftfreq(len(t), 1 / fs)[:len(t) // 2]

    line, = ax.plot(freqs, 20 * np.log10(spectrum / np.max(spectrum)))
    ax.set_ylim([-100, 5])
    ax.set_title('Widmo FFT sygnału sinusoidalnego')
    ax.grid(True)

    ax_freq = plt.axes([0.25, 0.15, 0.6, 0.03])
    s_freq = Slider(ax_freq, 'f [Hz]', 1, 500, valinit=init_f)

    def update(val=None):
        f = s_freq.val
        sig = np.sin(2 * np.pi * f * t)
        spectrum = np.abs(fft.fft(sig))[:len(t) // 2]
        amps = 20 * np.log10(spectrum / np.max(spectrum))
        line.set_ydata(amps)
        fig.canvas.draw_idle()

    s_freq.on_changed(update)

    plt.show()


def zadanie4():
    """FFT dla sygnału z pliku CSV"""
    file_path = input("Podaj ścieżkę do pliku CSV: ")
    path = Path(file_path)

    if not path.exists():
        print("Plik nie istnieje.")
        return

    try:
        df = pd.read_csv(path)
        sig = df.iloc[:, 0].to_numpy()

        spectrum = np.abs(fft.fft(sig))[:len(sig) // 2]
        freqs = np.fft.fftfreq(len(sig), 1)[:len(sig) // 2]

        plt.figure(figsize=(10, 5))
        plt.plot(freqs, 20 * np.log10(spectrum / np.max(spectrum)))
        plt.title(f'Widmo sygnału z pliku: {path.name}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda [dB]')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Błąd: {e}")


if __name__ == '__main__':
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
