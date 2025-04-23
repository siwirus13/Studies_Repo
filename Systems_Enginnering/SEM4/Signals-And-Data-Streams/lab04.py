import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d
from typing import Tuple
import numpy.typing as npt


def generate_signal(
    frequency: float, sampling_rate: float, duration: float = 1.0
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generuje ciągły i próbkowany sygnał sinusoidalny.

    Returns:
        t: Oś czasu sygnału ciągłego
        signal: Wartości sygnału ciągłego
        t_samples: Czas próbek
        samples: Próbkowane wartości sygnału
    """
    t = np.linspace(0, duration, 1000)
    signal = np.sin(2 * np.pi * frequency * t)

    t_samples = np.arange(0, duration, 1 / sampling_rate)
    samples = np.sin(2 * np.pi * frequency * t_samples)

    return t, signal, t_samples, samples


def plot_signal_with_sliders() -> None:
    """
    Uruchamia wykres z suwakami do wyboru częstotliwości i częstotliwości próbkowania.
    """
    init_f = 5.0
    init_fs = 20.0
    duration = 1.0

    t, signal, t_samples, samples = generate_signal(init_f, init_fs, duration)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    l1, = ax.plot(t, signal, label='Sygnał ciągły')
    l2, = ax.plot(t_samples, samples, 'o', label='Próbki')
    ax.set_title('Sygnał i próbki')
    ax.legend()
    ax.grid(True)

    ax_f = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_fs = plt.axes([0.25, 0.15, 0.65, 0.03])

    s_f = Slider(ax_f, 'f [Hz]', 1, 50, valinit=init_f)
    s_fs = Slider(ax_fs, 'fs [Hz]', 5, 100, valinit=init_fs)

    def update(val: float) -> None:
        f = s_f.val
        fs = s_fs.val
        t, signal, t_samples, samples = generate_signal(f, fs, duration)
        l1.set_ydata(signal)
        l2.set_xdata(t_samples)
        l2.set_ydata(samples)
        fig.canvas.draw_idle()

    s_f.on_changed(update)
    s_fs.on_changed(update)

    def proceed(event: object) -> None:
        plt.close()
        zadanie2(s_f.val, s_fs.val)

    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Dalej', color='lightblue', hovercolor='skyblue')
    button.on_clicked(proceed)

    plt.show()


def zadanie2(frequency: float, sampling_rate: float) -> None:
    """
    Wykonuje interpolację liniową na podstawie danych z zadania 1.
    """
    duration = 1.0
    t_full = np.linspace(0, duration, 1000)
    signal_full = np.sin(2 * np.pi * frequency * t_full)

    t_samples = np.arange(0, duration, 1 / sampling_rate)
    samples = np.sin(2 * np.pi * frequency * t_samples)

    interp_func = interp1d(t_samples, samples, kind='linear', fill_value='extrapolate')
    interpolated = interp_func(t_full)

    error = np.abs(signal_full - interpolated)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_full, signal_full, label='Sygnał oryginalny')
    plt.plot(t_full, interpolated, '--', label='Interpolacja liniowa')
    plt.title('Interpolacja liniowa')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_full, error, 'r')
    plt.title('Błąd interpolacji liniowej')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    zadanie3(frequency, sampling_rate)


def zadanie3(frequency: float, sampling_rate: float) -> None:
    """
    Interpolacja sygnału na podstawie równania Whittakera–Shannona.
    """
    duration = 1.0
    t_full = np.linspace(0, duration, 1000)
    signal_full = np.sin(2 * np.pi * frequency * t_full)

    T = 1 / sampling_rate
    t_samples = np.arange(0, duration, T)
    samples = np.sin(2 * np.pi * frequency * t_samples)

    def whittaker_shannon(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        reconstructed = np.zeros_like(t)
        for n, x_n in enumerate(samples):
            reconstructed += x_n * np.sinc((t - t_samples[n]) / T)
        return reconstructed

    interpolated_ws = whittaker_shannon(t_full)
    error_ws = np.abs(signal_full - interpolated_ws)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_full, signal_full, label='Sygnał oryginalny')
    plt.plot(t_full, interpolated_ws, '--', label='Interpolacja Whittakera-Shannona')
    plt.title('Interpolacja wg Whittakera–Shannona')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_full, error_ws, 'r')
    plt.title('Błąd interpolacji Whittakera–Shannona')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_signal_with_sliders()

