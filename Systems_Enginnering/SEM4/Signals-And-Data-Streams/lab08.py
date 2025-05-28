
from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import pandas as pd
import scipy.signal as sig


def generate_signal_csv():
    fs = 1000
    t = np.linspace(0, 1, fs)
    signal = 2 * np.sin(2 * np.pi * 30 * t) + np.cos(2 * np.pi * 90 * t)
    df = pd.DataFrame({'time': t, 'signal': signal})
    df.to_csv('signal_lab08.csv', index=False)


def zadanie1():
    fs = 1000
    t = np.linspace(0, 1, fs)

    signals = {
        "Sinus": lambda f: np.sin(2 * np.pi * f * t),
        "Prostokątny": lambda f: sig.square(2 * np.pi * f * t),
        "Piłokształtny": lambda f: sig.sawtooth(2 * np.pi * f * t),
        "Trójkątny": lambda f: 2 * np.abs(sig.sawtooth(2 * np.pi * f * t)) - 1,
        "Świergotliwy": lambda f: np.sin(2 * np.pi * t**2 * f),
        "Superpozycja": lambda f: np.sin(2 * np.pi * f * t) + np.cos(2 * np.pi * (f + 20) * t),
        "Impuls jednostkowy": lambda f: np.where(np.isclose(t, 0.5, atol=1/fs), 1.0, 0.0)
    }

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.3)
    current_label = ["Sinus"]
    current_freq = [10.0]
    signal = signals[current_label[0]](current_freq[0])
    spec = ax.specgram(signal, Fs=fs, cmap='plasma', scale='dB')
    ax.set_title("Spektrogram")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Częstotliwość [Hz]")

    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.5])
    radio = RadioButtons(ax_radio, list(signals.keys()))

    ax_slider = plt.axes([0.3, 0.2, 0.6, 0.03])
    slider = Slider(ax_slider, 'Częstotliwość', 1, 100, valinit=current_freq[0], valstep=1)

    def update_plot():
        label = current_label[0]
        f = current_freq[0]
        signal = signals[label](f)
        ax.clear()
        ax.specgram(signal, Fs=fs, cmap='plasma', scale='dB')
        ax.set_title("Spektrogram")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Częstotliwość [Hz]")
        fig.canvas.draw_idle()

    def on_radio_change(label):
        current_label[0] = label
        update_plot()

    def on_slider_change(val):
        current_freq[0] = val
        update_plot()

    radio.on_clicked(on_radio_change)
    slider.on_changed(on_slider_change)

    plt.show()


def zadanie2():
    fs = 1000
    t = np.linspace(0, 1, fs)
    fmax = 250

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.subplots_adjust(left=0.2, bottom=0.3)

    ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider = Slider(ax_slider, 'fmax', 10, 500, valinit=fmax, valstep=10)

    colorbar = [None]
    im = [None]

    def update(val):
        ax1.clear()
        ax2.clear()
        fmax_val = slider.val
        chirp = np.sin(2 * np.pi * t**2 * fmax_val)

        ax1.specgram(chirp, Fs=fs, cmap='plasma')
        ax1.set_title("Spektrogram sygnału Chirp")
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Częstotliwość [Hz]")

        emd = EMD()
        imfs = emd(chirp)

        N = imfs.shape[1]
        widmo = np.abs(np.fft.fft(imfs, axis=1))

        extent = [0, fs/2, 0, len(imfs)]
        img = ax2.imshow(widmo[:, :N//2], aspect='auto', cmap='plasma', extent=extent)
        ax2.set_title("Widma IMF – Chirp")
        ax2.set_ylabel("Mod")
        ax2.set_xlabel("Częstotliwość [Hz]")

        if colorbar[0] is not None:
            colorbar[0].remove()
        colorbar[0] = fig.colorbar(img, ax=ax2, label='Amplituda FFT')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(fmax)
    plt.show()


def zadanie3():
    fs = 1000
    t = np.linspace(0, 1, fs)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.25)

    amp1 = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'A1', 0, 5, valinit=3)
    amp2 = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'A2', 0, 5, valinit=2.5)

    colorbar = [None]

    def update(val):
        a1 = amp1.val
        a2 = amp2.val
        signal = a1 * np.sin(2 * np.pi * 30 * t) + a2 * np.cos(2 * np.pi * 60 * t) + 2 * np.sin(2 * np.pi * 90 * t)
        emd = EMD()
        imfs = emd(signal)

        ax.clear()
        N = imfs.shape[1]
        widmo = np.abs(np.fft.fft(imfs, axis=1))
        extent = [0, fs/2, 0, len(imfs)]
        img = ax.imshow(widmo[:, :N//2], aspect='auto', cmap='plasma', extent=extent)
        ax.set_title("Widma IMF – Superpozycja sinusów i cosinusów")
        ax.set_ylabel("Mod")
        ax.set_xlabel("Częstotliwość [Hz]")

        if colorbar[0] is not None:
            colorbar[0].remove()
        colorbar[0] = fig.colorbar(img, ax=ax, label='Amplituda FFT')
        fig.canvas.draw_idle()

    amp1.on_changed(update)
    amp2.on_changed(update)
    update(None)
    plt.show()


def zadanie4():
    df = pd.read_csv("signal_lab08.csv")
    t = df['time'].to_numpy()
    signal1 = df['signal'].to_numpy()
    fs = int(len(t) / (t[-1] - t[0]))

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.3)

    amp = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Wzmocnienie', 0.1, 5, valinit=1)

    colorbar = [None]

    def update(val):
        a = amp.val
        signal = signal1 + a * np.sin(2 * np.pi * 120 * t)
        emd = EMD()
        imfs = emd(signal)

        ax.clear()
        N = imfs.shape[1]
        widmo = np.abs(np.fft.fft(imfs, axis=1))
        extent = [0, fs/2, 0, len(imfs)]
        img = ax.imshow(widmo[:, :N//2], aspect='auto', cmap='plasma', extent=extent)
        ax.set_title("Widma IMF – Sygnał z pliku")
        ax.set_ylabel("Mod")
        ax.set_xlabel("Częstotliwość [Hz]")

        if colorbar[0] is not None:
            colorbar[0].remove()
        colorbar[0] = fig.colorbar(img, ax=ax, label='Amplituda FFT')
        fig.canvas.draw_idle()

    amp.on_changed(update)
    update(None)
    plt.show()


if __name__ == "__main__":
    generate_signal_csv()
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
