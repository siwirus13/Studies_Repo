import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.signal import chirp, square, sawtooth, hilbert

def generate_sine(f, fs, duration=1.0):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    return t, np.sin(2*np.pi*f*t)

def generate_by_type(sig_type, f, fs, duration=1.0):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    if sig_type == "Sinus":
        x = np.sin(2*np.pi*f*t)
    elif sig_type == "Prostokątny":
        x = square(2*np.pi*f*t)
    elif sig_type == "Piłokształtny":
        x = sawtooth(2*np.pi*f*t)
    elif sig_type == "Świergotliwy":
        x = chirp(t, f0=1, f1=f, t1=duration, method='linear')
    elif sig_type == "Sin + Cos":
        x = np.sin(2*np.pi*f*t) + 0.5*np.cos(2*np.pi*2*f*t)
    elif sig_type == "Impuls jednostkowy":
        x = np.zeros_like(t); x[0] = 1.0
    else:
        raise ValueError(sig_type)
    return t, x

def zad1_window():
    f0, fs0 = 10.0, 200.0
    fig, ax = plt.subplots(figsize=(6,4))
    plt.subplots_adjust(bottom=0.25)
    ln, = ax.plot([], [], lw=1.5)
    ax.set_title("Zadanie 1: Widmo amplitudowe (linear)")
    ax.set_xlabel("Hz"); ax.set_ylabel("Amp"); ax.grid(True)

    ax_f  = fig.add_axes([0.15, 0.10, 0.65, 0.03], facecolor='lightgray')
    ax_fs = fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgray')
    sf   = Slider(ax_f,  'f [Hz]', 1, 100, valinit=f0)
    sfs  = Slider(ax_fs, 'fs [Hz]', 50, 2000, valinit=fs0)

    def update(val=None):
        f, fs = sf.val, sfs.val
        t, x = generate_sine(f, fs)
        N = len(x); X = np.fft.fft(x); freq = np.fft.fftfreq(N,1/fs)
        half = slice(0, N//2)
        ln.set_data(freq[half], np.abs(X[half]))
        ax.set_xlim(0, fs/2)
        ax.set_ylim(0, np.abs(X[half]).max()*1.1)
        fig.canvas.draw_idle()

    sf.on_changed(update)
    sfs.on_changed(update)
    update()  # initial draw

def zad2_window():
    f0, fs0 = 10.0, 200.0
    fig, ax = plt.subplots(figsize=(6,4))
    plt.subplots_adjust(bottom=0.25)
    ln, = ax.plot([], [], lw=1.5)
    ax.set_title("Zadanie 2: Widmo amplitudowe (dB)")
    ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.grid(True)

    ax_f  = fig.add_axes([0.15, 0.10, 0.65, 0.03], facecolor='lightgray')
    ax_fs = fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgray')
    sf   = Slider(ax_f,  'f [Hz]', 1, 100, valinit=f0)
    sfs  = Slider(ax_fs, 'fs [Hz]', 50, 2000, valinit=fs0)

    def update(val=None):
        f, fs = sf.val, sfs.val
        t, x = generate_sine(f, fs)
        N = len(x); X = np.fft.fft(x); freq = np.fft.fftfreq(N,1/fs)
        half = slice(0, N//2)
        mag_db = 20*np.log10(np.abs(X[half])+1e-12)
        ln.set_data(freq[half], mag_db)
        ax.set_xlim(0, fs/2)
        ax.set_ylim(mag_db.min()*1.1, mag_db.max()*1.1)
        fig.canvas.draw_idle()

    sf.on_changed(update)
    sfs.on_changed(update)
    update()

def zad3_window():
    f0, fs0 = 10.0, 200.0
    types = ["Sinus","Prostokątny","Piłokształtny","Świergotliwy",
             "Sin + Cos","Impuls jednostkowy"]

    fig, ax = plt.subplots(figsize=(6,4))
    plt.subplots_adjust(left=0.30, bottom=0.25)
    ln, = ax.plot([], [], lw=1.5)
    ax.set_title("Zadanie 3: Widmo amplitudowe wybranego sygnału")
    ax.set_xlabel("Hz"); ax.set_ylabel("Amp"); ax.grid(True)

    ax_f  = fig.add_axes([0.30, 0.10, 0.60, 0.03], facecolor='lightgray')
    ax_fs = fig.add_axes([0.30, 0.05, 0.60, 0.03], facecolor='lightgray')
    ax_r  = fig.add_axes([0.05, 0.30, 0.20, 0.50], facecolor='lightgoldenrodyellow')

    sf   = Slider(ax_f,  'f [Hz]', 1, 100, valinit=f0)
    sfs  = Slider(ax_fs,'fs [Hz]', 50,2000, valinit=fs0)
    radio= RadioButtons(ax_r, types, active=0)

    def update(val=None):
        f, fs = sf.val, sfs.val
        sig  = radio.value_selected
        t, x = generate_by_type(sig, f, fs)
        N = len(x); X = np.fft.fft(x); freq = np.fft.fftfreq(N,1/fs)
        half = slice(0, N//2)
        ln.set_data(freq[half], np.abs(X[half]))
        ax.set_xlim(0, fs/2)
        ax.set_ylim(0, np.abs(X[half]).max()*1.1)
        fig.canvas.draw_idle()

    sf.on_changed(update)
    sfs.on_changed(update)
    radio.on_clicked(update)
    update()

def zad4_window():
    f0, fs0 = 10.0, 200.0
    fig, ax = plt.subplots(figsize=(6,4))
    plt.subplots_adjust(bottom=0.25)
    ln, = ax.plot([], [], lw=1.5)
    ax.set_title("Zadanie 4: Widmo fazowe (sinus)")
    ax.set_xlabel("Hz"); ax.set_ylabel("Phase [rad]"); ax.grid(True)

    ax_f  = fig.add_axes([0.15,0.10,0.65,0.03], facecolor='lightgray')
    ax_fs = fig.add_axes([0.15,0.05,0.65,0.03], facecolor='lightgray')
    sf   = Slider(ax_f, 'f [Hz]', 1,100, valinit=f0)
    sfs  = Slider(ax_fs,'fs [Hz]', 50,2000, valinit=fs0)

    def update(val=None):
        f, fs = sf.val, sfs.val
        t, x = generate_sine(f, fs)
        N = len(x); X = np.fft.fft(x); freq = np.fft.fftfreq(N,1/fs)
        half = slice(0, N//2)
        phase = np.angle(X[half])
        ln.set_data(freq[half], phase)
        ax.set_xlim(0, fs/2)
        ax.set_ylim(phase.min()*1.1, phase.max()*1.1)
        fig.canvas.draw_idle()

    sf.on_changed(update)
    sfs.on_changed(update)
    update()

def zad5_window():
    fs0, f1_0 = 200.0, 100.0
    fig, ax = plt.subplots(figsize=(6,4))
    plt.subplots_adjust(bottom=0.25)
    ln_sig, = ax.plot([], [], label="Chirp")
    ln_env, = ax.plot([], [], 'r--', label="Envelope")
    ax.set_title("Zadanie 5: Obwiednia sygnału świergotliwego")
    ax.set_xlabel("Time [s]"); ax.legend(); ax.grid(True)

    ax_fs = fig.add_axes([0.15,0.10,0.65,0.03], facecolor='lightgray')
    ax_f1 = fig.add_axes([0.15,0.05,0.65,0.03], facecolor='lightgray')
    sfs = Slider(ax_fs, 'fs [Hz]', 50,2000, valinit=fs0)
    sf1 = Slider(ax_f1, 'f1 [Hz]', 1,500, valinit=f1_0)

    def update(val=None):
        fs, f1 = sfs.val, sf1.val
        duration = 1.0
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        x = chirp(t, f0=1, f1=f1, t1=duration, method='linear')
        env = np.abs(hilbert(x))
        ln_sig.set_data(t, x)
        ln_env.set_data(t, env)
        ax.set_xlim(0, duration)
        ymin, ymax = min(x.min(), env.min()), max(x.max(), env.max())
        ax.set_ylim(ymin*1.1, ymax*1.1)
        fig.canvas.draw_idle()

    sfs.on_changed(update)
    sf1.on_changed(update)
    update()

if __name__ == "__main__":
    zad1_window()
    zad2_window()
    zad3_window()
    zad4_window()
    zad5_window()
    plt.show()

