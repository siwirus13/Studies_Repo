import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import chirp
import matplotlib.pyplot as plt
import pywt
from scipy.signal import chirp
from matplotlib.widgets import RadioButtons, Slider


def zadanie1():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider
    import numpy as np
    import pywt
    from scipy.special import hermite

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.25)
    ax.set_title('Typ falki')

    # Falki działające z wavefun + ręczne
    wavelet_options = {
        "Haar": ("haar", "discrete_fixed", None),
        "Daubechies": ("db", "discrete_param", (1, 20)),
        "Symlets": ("sym", "discrete_param", (1, 20)),
        "Coiflets": ("coif", "discrete_param", (1, 5)),
        "Biorthogonal": ("bior", "discrete_param", (1.1, 3.9)),  # tylko te, które działają
        "Mexican hat": ("mexh", "continuous", (0.1, 5.0)),
        "Morlet": ("morl", "continuous", (0.1, 5.0)),
        "Gaussian (manual)": ("gaussian_manual", "manual_param", (1, 6)),
        "Shannon (manual)": ("shannon_manual", "manual", None),
    }

    default_label = "Morlet"

    def gaussian_wavelet(order, x):
        H = hermite(order)
        return (-1)**order * H(x) * np.exp(-x**2 / 2)

    def shannon_wavelet(x):
        return np.sinc(x) * np.cos(2 * np.pi * x)

    def compute_wavelet(wavelet_base, wavelet_type, value):
        try:
            if wavelet_type == "discrete_fixed":
                wavelet = pywt.Wavelet(wavelet_base)
                _, psi, x = wavelet.wavefun()
                return x, psi
            elif wavelet_type == "discrete_param":
                if wavelet_base == "bior":
                    value = round(value * 10) / 10
                    wid = f"bior{value:.1f}".replace(".", ".")
                else:
                    wid = f"{wavelet_base}{int(value)}"
                wavelet = pywt.Wavelet(wid)
                _, psi, x = wavelet.wavefun()
                return x, psi
            elif wavelet_type == "continuous":
                wavelet = pywt.ContinuousWavelet(wavelet_base)
                psi, x = wavelet.wavefun(level=10)
                x_scaled = x * value
                psi_scaled = psi / np.sqrt(value)
                return x_scaled, psi_scaled
            elif wavelet_type == "manual_param" and wavelet_base == "gaussian_manual":
                x = np.linspace(-5, 5, 500)
                y = gaussian_wavelet(int(value), x)
                return x, y
            elif wavelet_type == "manual" and wavelet_base == "shannon_manual":
                x = np.linspace(-5, 5, 500)
                y = shannon_wavelet(x)
                return x, y
        except Exception:
            return np.linspace(-1, 1, 200), np.zeros(200)

    # Inicjalizacja
    wavelet_base, wavelet_type, param_range = wavelet_options[default_label]
    param_init = 1.0 if wavelet_type not in ("discrete_fixed", "manual") else None
    x, psi = compute_wavelet(wavelet_base, wavelet_type, param_init)
    l, = ax.plot(x, psi, label=default_label)
    ax.legend()

    # Przyciski i suwak
    ax_radio = plt.axes([0.05, 0.3, 0.2, 0.6])
    radio = RadioButtons(ax_radio, list(wavelet_options.keys()))

    ax_slider = plt.axes([0.3, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Parametr', 0.1, 5.0, valinit=param_init or 1.0, valstep=0.1)

    def update_slider(label):
        _, wtype, prange = wavelet_options[label]
        if wtype in ("discrete_fixed", "manual"):
            ax_slider.set_visible(False)
        else:
            ax_slider.set_visible(True)
            slider.label.set_text("Numer" if "param" in wtype else "Skala (Hz⁻¹)")
            slider.valmin, slider.valmax = prange
            slider.ax.set_xlim(prange)
            slider.set_val((prange[0] + prange[1]) / 2)

    def update_plot(label=None):
        if label is None:
            label = radio.value_selected
        base, wtype, _ = wavelet_options[label]
        param = slider.val if wtype not in ("discrete_fixed", "manual") else None
        x_new, psi_new = compute_wavelet(base, wtype, param)
        l.set_xdata(x_new)
        l.set_ydata(psi_new)
        l.set_label(f"{label} ({param:.2f})" if param else label)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()

    def on_radio_change(label):
        update_slider(label)
        update_plot(label)

    radio.on_clicked(on_radio_change)
    slider.on_changed(lambda val: update_plot())

    update_slider(default_label)
    update_plot(default_label)
    plt.show()



def zadanie2():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.set_title('Falka Daubechies')

    db_level = 1
    wavelet = pywt.Wavelet(f'db{db_level}')
    phi, psi, x = wavelet.wavefun()
    l_phi, = ax.plot(x, phi, label='φ (skalująca)')
    l_psi, = ax.plot(x, psi, label='ψ (falkowa)')
    ax.legend()

    ax_slider = plt.axes([0.25, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'db', 1, 20, valinit=1, valstep=1)

    def update(val):
        level = int(slider.val)
        try:
            wavelet = pywt.Wavelet(f'db{level}')
            phi, psi, x = wavelet.wavefun()
            l_phi.set_data(x, phi)
            l_psi.set_data(x, psi)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error: {e}")

    slider.on_changed(update)
    plt.show()


def zadanie3():
    def generate_chirp():
        t = np.linspace(0, 1, 1024)
        return chirp(t, f0=6, f1=1, t1=1, method='linear')

    def generate_sine():
        t = np.linspace(0, 1, 1024)
        return np.sin(2 * np.pi * 15 * t)

    signal_generators = {
        "Chirp": generate_chirp,
        "Sine": generate_sine,
    }

    wavelets = ["db4", "sym4", "coif1"]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.25, hspace=0.6)

    signal = generate_chirp()
    axs[0].plot(signal)
    axs[0].set_title("Sygnał wejściowy")

    coeffs_list = []
    for i, wavelet in enumerate(wavelets):
        coeffs = pywt.wavedec(signal, wavelet, level=3)
        coeffs_list.append(coeffs)
        axs[i + 1].plot(np.concatenate(coeffs))
        axs[i + 1].set_title(f'Dekompozycja falką {wavelet}')

    ax_radio = plt.axes([0.05, 0.5, 0.15, 0.3])
    radio = RadioButtons(ax_radio, list(signal_generators.keys()))

    def update(label):
        new_signal = signal_generators[label]()
        axs[0].cla()
        axs[0].plot(new_signal)
        axs[0].set_title("Sygnał wejściowy")

        for i, wavelet in enumerate(wavelets):
            coeffs = pywt.wavedec(new_signal, wavelet, level=3)
            axs[i + 1].cla()
            axs[i + 1].plot(np.concatenate(coeffs))
            axs[i + 1].set_title(f'Dekompozycja falką {wavelet}')

        fig.canvas.draw_idle()

    radio.on_clicked(update)
    plt.show()


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
