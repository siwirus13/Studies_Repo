import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import interp1d
from scipy.signal import square, sawtooth
from typing import Tuple
import numpy.typing as npt


def generate_signal(
    frequency: float, sampling_rate: float, duration: float = 1.0, signal_type: str = 'sin'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generuje ciągły i próbkowany sygnał: sinus, prostokątny lub trójkątny.

    Returns:
        t: Oś czasu sygnału ciągłego
        signal: Wartości sygnału ciągłego
        t_samples: Czas próbek
        samples: Próbkowane wartości sygnału
    """
    t = np.linspace(0, duration, 1000)
    t_samples = np.arange(0, duration, 1 / sampling_rate)

    if signal_type == 'sin':
        signal = np.sin(2 * np.pi * frequency * t)
        samples = np.sin(2 * np.pi * frequency * t_samples)
    elif signal_type == 'square':
        signal = square(2 * np.pi * frequency * t)
        samples = square(2 * np.pi * frequency * t_samples)
    elif signal_type == 'triangle':
        signal = sawtooth(2 * np.pi * frequency * t, width=0.5)
        samples = sawtooth(2 * np.pi * frequency * t_samples, width=0.5)
    else:
        raise ValueError(f"Nieznany typ sygnału: {signal_type}")

    return t, signal, t_samples, samples


def plot_signal_with_sliders() -> None:
    """
    Zadanie 1: Wykres z suwakami częstotliwości i próbkowania oraz wyborem typu sygnału.
    """
    init_f = 5.0
    init_fs = 20.0
    init_signal_type = 'sin'
    duration = 1.0

    t, signal, t_samples, samples = generate_signal(init_f, init_fs, duration, init_signal_type)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.35, bottom=0.35)
    l1, = ax.plot(t, signal, label='Sygnał ciągły')
    l2, = ax.plot(t_samples, samples, 'o', label='Próbki')
    ax.set_title('Sygnał i próbki')
    ax.legend()
    ax.grid(True)

    ax_f = plt.axes([0.35, 0.25, 0.55, 0.03])
    ax_fs = plt.axes([0.35, 0.2, 0.55, 0.03])

    s_f = Slider(ax_f, 'f [Hz]', 1, 50, valinit=init_f)
    s_fs = Slider(ax_fs, 'fs [Hz]', 5, 100, valinit=init_fs)

    radio_ax = plt.axes([0.05, 0.5, 0.2, 0.15], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(radio_ax, ('sin', 'square', 'triangle'))

    current_signal_type = [init_signal_type]

    def update(val: float = None) -> None:
        f = s_f.val
        fs = s_fs.val
        signal_type = current_signal_type[0]
        t, signal, t_samples, samples = generate_signal(f, fs, duration, signal_type)
        l1.set_ydata(signal)
        l2.set_xdata(t_samples)
        l2.set_ydata(samples)
        fig.canvas.draw_idle()

    s_f.on_changed(update)
    s_fs.on_changed(update)

    def change_signal_type(label: str) -> None:
        current_signal_type[0] = label
        update()

    radio.on_clicked(change_signal_type)

    def proceed(event: object) -> None:
        plt.close()
        zadanie2(s_f.val, s_fs.val, current_signal_type[0])

    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Dalej', color='lightblue', hovercolor='skyblue')
    button.on_clicked(proceed)

    plt.show()


def zadanie2(frequency: float, sampling_rate: float, signal_type: str) -> None:
    """
    Zadanie 2: Interpolacja liniowa sygnału.
    """
    duration = 1.0
    t_full = np.linspace(0, duration, 1000)

    if signal_type == 'sin':
        signal_full = np.sin(2 * np.pi * frequency * t_full)
    elif signal_type == 'square':
        signal_full = square(2 * np.pi * frequency * t_full)
    elif signal_type == 'triangle':
        signal_full = sawtooth(2 * np.pi * frequency * t_full, width=0.5)
    else:
        raise ValueError(f"Nieznany typ sygnału: {signal_type}")

    t_samples = np.arange(0, duration, 1 / sampling_rate)
    if signal_type == 'sin':
        samples = np.sin(2 * np.pi * frequency * t_samples)
    elif signal_type == 'square':
        samples = square(2 * np.pi * frequency * t_samples)
    elif signal_type == 'triangle':
        samples = sawtooth(2 * np.pi * frequency * t_samples, width=0.5)

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

    zadanie3(frequency, sampling_rate, signal_type)


def zadanie3(frequency: float, sampling_rate: float, signal_type: str) -> None:
    """
    Zadanie 3: Interpolacja sygnału wg. Whittakera–Shannona.
    """
    duration = 1.0
    t_full = np.linspace(0, duration, 1000)

    if signal_type == 'sin':
        signal_full = np.sin(2 * np.pi * frequency * t_full)
    elif signal_type == 'square':
        signal_full = square(2 * np.pi * frequency * t_full)
    elif signal_type == 'triangle':
        signal_full = sawtooth(2 * np.pi * frequency * t_full, width=0.5)
    else:
        raise ValueError(f"Nieznany typ sygnału: {signal_type}")

    T = 1 / sampling_rate
    t_samples = np.arange(0, duration, T)
    if signal_type == 'sin':
        samples = np.sin(2 * np.pi * frequency * t_samples)
    elif signal_type == 'square':
        samples = square(2 * np.pi * frequency * t_samples)
    elif signal_type == 'triangle':
        samples = sawtooth(2 * np.pi * frequency * t_samples, width=0.5)

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
    plt.plot(t_full, interpolated_ws, '--', label='Interpolacja Whittakera–Shannona')
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

