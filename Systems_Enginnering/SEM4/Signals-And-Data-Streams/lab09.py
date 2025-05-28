import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
from scipy.signal import savgol_filter
import pandas as pd
from PyEMD import EMD
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def zadanie1():
    """Implementacja miar jakości sygnałów: SNR, PSNR, MSE"""
    
    def calculate_snr(signal_clean, noise):
        """SNR = 20*log10(s/n)"""
        signal_power = np.sqrt(np.mean(signal_clean**2))
        noise_power = np.sqrt(np.mean(noise**2))
        if noise_power == 0:
            return float('inf')
        return 20 * np.log10(signal_power / noise_power)
    
    def calculate_psnr(signal_clean, signal_noisy):
        """PSNR = 20*log10(s_max/sqrt(MSE))"""
        s_max = np.max(np.abs(signal_clean))
        mse = np.mean((signal_clean - signal_noisy)**2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(s_max / np.sqrt(mse))
    
    def calculate_mse(signal_clean, signal_noisy):
        """MSE = (1/N) * sum((s_n - y_n)^2)"""
        return np.mean((signal_clean - signal_noisy)**2)
    
    # Generowanie sygnału testowego
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), False)
    
    # Tworzenie interfejsu
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Suwaki
    ax_freq = plt.axes([0.1, 0.08, 0.35, 0.03])
    ax_noise = plt.axes([0.55, 0.08, 0.35, 0.03])
    
    slider_freq = Slider(ax_freq, 'Częstotliwość [Hz]', 10, 200, valinit=50)
    slider_noise = Slider(ax_noise, 'Poziom szumu', 0.01, 2.0, valinit=0.3)
    
    def update_metrics(val):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        freq = slider_freq.val
        noise_level = slider_noise.val
        
        # Generowanie sygnałów
        clean_signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.randn(len(t)) * noise_level
        noisy_signal = clean_signal + noise
        
        # Obliczanie metryk
        snr_val = calculate_snr(clean_signal, noise)
        psnr_val = calculate_psnr(clean_signal, noisy_signal)
        mse_val = calculate_mse(clean_signal, noisy_signal)
        
        # Wykresy
        ax1.plot(t, clean_signal, 'b-', linewidth=2, label='Sygnał czysty')
        ax1.set_title('Sygnał referencyjny')
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Amplituda')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(t, noisy_signal, 'r-', alpha=0.7, label='Sygnał z szumem')
        ax2.set_title('Sygnał zaszumiony')
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('Amplituda')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(t, noise, 'g-', alpha=0.7, label='Szum')
        ax3.set_title('Szum')
        ax3.set_xlabel('Czas [s]')
        ax3.set_ylabel('Amplituda')
        ax3.legend()
        ax3.grid(True)
        
        # Wyświetlanie metryk
        metrics_text = f'SNR = {snr_val:.2f} dB\nPSNR = {psnr_val:.2f} dB\nMSE = {mse_val:.6f}'
        ax4.text(0.1, 0.7, metrics_text, fontsize=16, transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Miary jakości sygnału')
        ax4.axis('off')
        
        # Histogram porównania
        ax4.hist(clean_signal, bins=30, alpha=0.5, label='Czysty', color='blue')
        ax4.hist(noisy_signal, bins=30, alpha=0.5, label='Zaszumiony', color='red')
        ax4.legend()
        ax4.set_xlabel('Amplituda')
        ax4.set_ylabel('Liczba próbek')
        
        plt.tight_layout()
        plt.draw()
    
    slider_freq.on_changed(update_metrics)
    slider_noise.on_changed(update_metrics)
    update_metrics(None)
    plt.show()

def zadanie2():
    """Porównanie własnych implementacji z gotowymi bibliotekami"""
    
    def calculate_snr_custom(signal_clean, noise):
        """Własna implementacja SNR"""
        signal_power = np.sqrt(np.mean(signal_clean**2))
        noise_power = np.sqrt(np.mean(noise**2))
        if noise_power == 0:
            return float('inf')
        return 20 * np.log10(signal_power / noise_power)
    
    def calculate_psnr_custom(signal_clean, signal_noisy):
        """Własna implementacja PSNR"""
        s_max = np.max(np.abs(signal_clean))
        mse = np.mean((signal_clean - signal_noisy)**2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(s_max / np.sqrt(mse))
    
    def calculate_mse_custom(signal_clean, signal_noisy):
        """Własna implementacja MSE"""
        return np.mean((signal_clean - signal_noisy)**2)
    
    # Generowanie sygnału testowego
    fs = 1000
    T = 1.0
    t = np.linspace(0, T, int(fs * T), False)
    
    # Tworzenie interfejsu
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Suwaki
    ax_amp = plt.axes([0.1, 0.08, 0.35, 0.03])
    ax_noise = plt.axes([0.55, 0.08, 0.35, 0.03])
    
    slider_amp = Slider(ax_amp, 'Amplituda sygnału', 0.1, 5.0, valinit=1.0)
    slider_noise = Slider(ax_noise, 'Poziom szumu', 0.01, 1.0, valinit=0.2)
    
    def update_comparison(val):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        amplitude = slider_amp.val
        noise_level = slider_noise.val
        
        # Generowanie sygnałów
        clean_signal = amplitude * np.sin(2 * np.pi * 50 * t)
        noise = np.random.randn(len(t)) * noise_level
        noisy_signal = clean_signal + noise
        
        # Normalizacja dla skimage (0-1)
        clean_norm = (clean_signal - np.min(clean_signal)) / (np.max(clean_signal) - np.min(clean_signal))
        noisy_norm = (noisy_signal - np.min(noisy_signal)) / (np.max(noisy_signal) - np.min(noisy_signal))
        
        # Własne implementacje
        snr_custom = calculate_snr_custom(clean_signal, noise)
        psnr_custom = calculate_psnr_custom(clean_signal, noisy_signal)
        mse_custom = calculate_mse_custom(clean_signal, noisy_signal)
        
        # Biblioteki zewnętrzne
        try:
            psnr_skimage = peak_signal_noise_ratio(clean_norm, noisy_norm)
            mse_skimage = mean_squared_error(clean_norm, noisy_norm)
        except:
            psnr_skimage = "N/A"
            mse_skimage = "N/A"
        
        # Wykresy
        ax1.plot(t, clean_signal, 'b-', linewidth=2, label='Sygnał czysty')
        ax1.plot(t, noisy_signal, 'r-', alpha=0.7, label='Sygnał zaszumiony')
        ax1.set_title('Porównanie sygnałów')
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Amplituda')
        ax1.legend()
        ax1.grid(True)
        
        # Różnica między sygnałami
        difference = clean_signal - noisy_signal
        ax2.plot(t, difference, 'g-', linewidth=2, label='Różnica (błąd)')
        ax2.set_title('Błąd rekonstrukcji')
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('Amplituda')
        ax2.legend()
        ax2.grid(True)
        
        # Porównanie metryk - własne vs biblioteki
        categories = ['MSE', 'PSNR']
        custom_values = [mse_custom, psnr_custom]
        library_values = [mse_skimage if mse_skimage != "N/A" else 0, 
                         psnr_skimage if psnr_skimage != "N/A" else 0]
        
        x = np.arange(len(categories))
        width = 0.35
        
        if mse_skimage != "N/A" and psnr_skimage != "N/A":
            ax3.bar(x - width/2, custom_values, width, label='Własna implementacja', alpha=0.7)
            ax3.bar(x + width/2, library_values, width, label='Biblioteka (skimage)', alpha=0.7)
        else:
            ax3.bar(x, custom_values, width, label='Własna implementacja', alpha=0.7)
        
        ax3.set_title('Porównanie metryk')
        ax3.set_xlabel('Metryki')
        ax3.set_ylabel('Wartość')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True)
        
        # Tabela wyników
        results_text = f"""WYNIKI PORÓWNANIA:
        
Własne implementacje:
• SNR: {snr_custom:.3f} dB
• PSNR: {psnr_custom:.3f} dB  
• MSE: {mse_custom:.6f}

Biblioteki zewnętrzne:
• PSNR (skimage): {psnr_skimage:.3f if psnr_skimage != 'N/A' else 'N/A'}
• MSE (skimage): {mse_skimage:.6f if mse_skimage != 'N/A' else 'N/A'}

Różnice:
• PSNR: {abs(psnr_custom - psnr_skimage):.6f if psnr_skimage != 'N/A' else 'N/A'}
• MSE: {abs(mse_custom - mse_skimage):.6f if mse_skimage != 'N/A' else 'N/A'}"""
        
        ax4.text(0.05, 0.95, results_text, fontsize=10, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax4.set_title('Szczegółowe porównanie')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.draw()
    
    slider_amp.on_changed(update_comparison)
    slider_noise.on_changed(update_comparison)
    update_comparison(None)
    plt.show()

def zadanie3():
    """Generowanie sygnału świergotliwego z szumem białym i brązowym"""
    
    # Parametry
    fs = 1000  # częstotliwość próbkowania
    T = 2.0    # czas trwania sygnału
    t = np.linspace(0, T, int(fs * T), False)
    
    # Tworzenie interfejsu z suwakiem
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Suwak dla SNR
    ax_snr = plt.axes([0.2, 0.02, 0.5, 0.03])
    slider_snr = Slider(ax_snr, 'SNR [dB]', -10, 20, valinit=5, valfmt='%d')
    
    def update_signal(snr_db):
        # Czyszczenie wykresów
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Sygnał świergotliwy (chirp)
        f0, f1 = 50, 400  # częstotliwości początkowa i końcowa
        chirp_signal = signal.chirp(t, f0, T, f1, method='linear')
        
        # Szum biały
        white_noise = np.random.randn(len(t))
        
        # Szum brązowy (brown noise)
        brown_noise = np.cumsum(np.random.randn(len(t)))
        brown_noise = brown_noise / np.std(brown_noise)  # normalizacja
        
        # Obliczanie mocy sygnału i szumu
        signal_power = np.mean(chirp_signal**2)
        noise_power_target = signal_power / (10**(snr_db/10))
        
        # Skalowanie szumów
        white_noise_scaled = white_noise * np.sqrt(noise_power_target)
        brown_noise_scaled = brown_noise * np.sqrt(noise_power_target)
        
        # Sygnały z szumem
        signal_with_white = chirp_signal + white_noise_scaled
        signal_with_brown = chirp_signal + brown_noise_scaled
        
        # Wykresy
        ax1.plot(t, chirp_signal, 'b-', linewidth=2)
        ax1.set_title('Sygnał świergotliwy (czysty)')
        ax1.set_ylabel('Amplituda')
        ax1.grid(True)
        
        ax2.plot(t, signal_with_white, 'r-', alpha=0.7)
        ax2.set_title(f'Sygnał z szumem białym (SNR = {snr_db} dB)')
        ax2.set_ylabel('Amplituda')
        ax2.grid(True)
        
        ax3.plot(t, signal_with_brown, 'g-', alpha=0.7)
        ax3.set_title(f'Sygnał z szumem brązowym (SNR = {snr_db} dB)')
        ax3.set_ylabel('Amplituda')
        ax3.grid(True)
        
        # Spektrogram
        f, t_spec, Sxx = signal.spectrogram(signal_with_white, fs)
        ax4.pcolormesh(t_spec, f, 10*np.log10(Sxx))
        ax4.set_title('Spektrogram sygnału z szumem białym')
        ax4.set_ylabel('Częstotliwość [Hz]')
        ax4.set_xlabel('Czas [s]')
        
        plt.tight_layout()
        plt.draw()
    
    slider_snr.on_changed(update_signal)
    update_signal(5)  # wartość początkowa
    plt.show()

def zadanie4():
    """Odszumanie sygnału filtrem Wienera"""
    
    # Generowanie sygnału testowego
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), False)
    f0, f1 = 50, 400
    clean_signal = signal.chirp(t, f0, T, f1, method='linear')
    
    # Tworzenie interfejsu
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Suwak dla długości filtru
    ax_filter = plt.axes([0.2, 0.02, 0.5, 0.03])
    slider_filter = Slider(ax_filter, 'Długość filtru', 10, 200, valinit=50, valfmt='%d')
    
    def wiener_filter(noisy_signal, filter_length):
        """Implementacja filtru Wienera"""
        N = len(noisy_signal)
        # Estymacja PSD sygnału i szumu
        f, Pxx_noisy = signal.welch(noisy_signal, fs, nperseg=min(256, N//4))
        
        # Prosta estymacja - założenie że wysokie częstotliwości to głównie szum
        noise_floor = np.median(Pxx_noisy[-len(Pxx_noisy)//4:])
        Pxx_signal = np.maximum(Pxx_noisy - noise_floor, 0.1 * Pxx_noisy)
        
        # Funkcja przejścia Wienera
        H = Pxx_signal / (Pxx_signal + noise_floor)
        
        # Filtracja w dziedzinie częstotliwości
        fft_signal = np.fft.fft(noisy_signal)
        freqs = np.fft.fftfreq(N, 1/fs)
        
        # Interpolacja H do wszystkich częstotliwości
        H_interp = np.interp(np.abs(freqs[:N//2]), f, H)
        H_full = np.concatenate([H_interp, H_interp[::-1]])
        
        filtered_fft = fft_signal * H_full
        return np.real(np.fft.ifft(filtered_fft))
    
    def update_filter(filter_len):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Dodanie szumu
        noise = np.random.randn(len(t)) * 0.3
        noisy_signal = clean_signal + noise
        
        # Filtracja Wienera
        filtered_signal = wiener_filter(noisy_signal, int(filter_len))
        
        # Wykresy
        ax1.plot(t, clean_signal, 'b-', label='Sygnał czysty', linewidth=2)
        ax1.set_title('Sygnał referencyjny')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(t, noisy_signal, 'r-', alpha=0.7, label='Sygnał z szumem')
        ax2.set_title('Sygnał zaszumiony')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(t, filtered_signal, 'g-', label='Po filtrze Wienera', linewidth=2)
        ax3.set_title(f'Sygnał po filtracji (długość filtru: {int(filter_len)})')
        ax3.legend()
        ax3.grid(True)
        
        # Porównanie spektrogramów
        f_spec, t_spec, Sxx_clean = signal.spectrogram(clean_signal, fs)
        f_spec, t_spec, Sxx_filtered = signal.spectrogram(filtered_signal, fs)
        
        im = ax4.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_filtered))
        ax4.set_title('Spektrogram sygnału odfiltrowanego')
        ax4.set_xlabel('Czas [s]')
        ax4.set_ylabel('Częstotliwość [Hz]')
        
        plt.tight_layout()
        plt.draw()
    
    slider_filter.on_changed(update_filter)
    update_filter(50)
    plt.show()

def zadanie5():
    """Odszumanie sygnału filtrem Savitzky-Golay"""
    
    # Generowanie sygnału testowego
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), False)
    f0, f1 = 50, 400
    clean_signal = signal.chirp(t, f0, T, f1, method='linear')
    noise = np.random.randn(len(t)) * 0.3
    noisy_signal = clean_signal + noise
    
    # Tworzenie interfejsu
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Suwak dla długości okna
    ax_window = plt.axes([0.2, 0.02, 0.5, 0.03])
    slider_window = Slider(ax_window, 'Długość okna', 5, 101, valinit=21, valfmt='%d')
    
    def update_savgol(window_length):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Zapewnienie nieparzystej długości okna
        win_len = int(window_length)
        if win_len % 2 == 0:
            win_len += 1
        
        # Filtracja Savitzky-Golay
        poly_order = min(3, win_len-1)
        filtered_signal = savgol_filter(noisy_signal, win_len, poly_order)
        
        # Obliczanie błędu
        mse = np.mean((clean_signal - filtered_signal)**2)
        
        # Wykresy
        ax1.plot(t, clean_signal, 'b-', label='Sygnał czysty', linewidth=2)
        ax1.set_title('Sygnał referencyjny')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(t, noisy_signal, 'r-', alpha=0.7, label='Sygnał z szumem')
        ax2.set_title('Sygnał zaszumiony')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(t, filtered_signal, 'g-', label='Po filtrze S-G', linewidth=2)
        ax3.set_title(f'Filtr Savitzky-Golay (okno: {win_len}, MSE: {mse:.4f})')
        ax3.legend()
        ax3.grid(True)
        
        # Porównanie w dziedzinie częstotliwości
        freqs = np.fft.fftfreq(len(t), 1/fs)
        fft_clean = np.abs(np.fft.fft(clean_signal))
        fft_filtered = np.abs(np.fft.fft(filtered_signal))
        
        ax4.plot(freqs[:len(freqs)//2], fft_clean[:len(freqs)//2], 'b-', label='Czysty')
        ax4.plot(freqs[:len(freqs)//2], fft_filtered[:len(freqs)//2], 'g-', label='Filtrowany')
        ax4.set_title('Porównanie widm')
        ax4.set_xlabel('Częstotliwość [Hz]')
        ax4.set_ylabel('Amplituda')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.draw()
    
    slider_window.on_changed(update_savgol)
    update_savgol(21)
    plt.show()

def zadanie6():
    """Odszumanie sygnału metodą EMD z częściową rekonstrukcją"""
    
    # Generowanie sygnału testowego
    fs = 1000
    T = 2.0
    t = np.linspace(0, T, int(fs * T), False)
    f0, f1 = 50, 400
    clean_signal = signal.chirp(t, f0, T, f1, method='linear')
    noise = np.random.randn(len(t)) * 0.3
    noisy_signal = clean_signal + noise
    
    # Tworzenie interfejsu
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Suwak dla liczby IMF do rekonstrukcji
    ax_imf = plt.axes([0.2, 0.02, 0.5, 0.03])
    slider_imf = Slider(ax_imf, 'Liczba IMF', 1, 8, valinit=4, valfmt='%d')
    
    def update_emd(num_imfs):
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        # Dekompozycja EMD
        emd = EMD()
        IMFs = emd.emd(noisy_signal)
        
        # Rekonstrukcja z wybraną liczbą IMF
        num_imfs = min(int(num_imfs), len(IMFs))
        reconstructed = np.sum(IMFs[:num_imfs], axis=0)
        
        # Obliczanie błędu
        mse = np.mean((clean_signal - reconstructed)**2)
        
        # Wykresy
        axes[0,0].plot(t, clean_signal, 'b-', linewidth=2)
        axes[0,0].set_title('Sygnał referencyjny')
        axes[0,0].grid(True)
        
        axes[0,1].plot(t, noisy_signal, 'r-', alpha=0.7)
        axes[0,1].set_title('Sygnał zaszumiony')
        axes[0,1].grid(True)
        
        axes[1,0].plot(t, reconstructed, 'g-', linewidth=2)
        axes[1,0].set_title(f'Rekonstrukcja EMD ({num_imfs} IMF, MSE: {mse:.4f})')
        axes[1,0].grid(True)
        
        # Porównanie wszystkich sygnałów
        axes[1,1].plot(t, clean_signal, 'b-', label='Czysty', linewidth=2)
        axes[1,1].plot(t, noisy_signal, 'r-', alpha=0.5, label='Zaszumiony')
        axes[1,1].plot(t, reconstructed, 'g-', label='EMD', linewidth=2)
        axes[1,1].set_title('Porównanie sygnałów')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Spektrogramy
        f_spec, t_spec, Sxx_clean = signal.spectrogram(clean_signal, fs, nperseg=128)
        f_spec, t_spec, Sxx_recon = signal.spectrogram(reconstructed, fs, nperseg=128)
        
        axes[2,0].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_clean))
        axes[2,0].set_title('Spektrogram - sygnał czysty')
        axes[2,0].set_xlabel('Czas [s]')
        axes[2,0].set_ylabel('Częstotliwość [Hz]')
        
        axes[2,1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_recon))
        axes[2,1].set_title('Spektrogram - rekonstrukcja EMD')
        axes[2,1].set_xlabel('Czas [s]')
        axes[2,1].set_ylabel('Częstotliwość [Hz]')
        
        plt.tight_layout()
        plt.draw()
    
    slider_imf.on_changed(update_emd)
    update_emd(4)
    plt.show()




def zadanie7():
    """Odszumanie sygnału z pliku CSV wszystkimi metodami"""
    
    # Tworzenie przykładowego pliku CSV jeśli nie istnieje
    csv_filename = 'test_signal.csv'
    try:
        data = pd.read_csv(csv_filename)
        if 'signal' not in data.columns:
            raise FileNotFoundError
        signal_data = data['signal'].values
    except FileNotFoundError:
        # Generowanie przykładowych danych
        fs = 1000
        T = 3.0
        t = np.linspace(0, T, int(fs * T), False)
        # Kombinacja sygnałów sinusoidalnych z szumem
        clean = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.3*np.sin(2*np.pi*200*t)
        noise = np.random.randn(len(t)) * 0.4
        signal_data = clean + noise
        
        # Zapisanie do CSV
        df = pd.DataFrame({'time': t, 'signal': signal_data})
        df.to_csv(csv_filename, index=False)
        print(f"Utworzono przykładowy plik: {csv_filename}")
    
    # Tworzenie interfejsu
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Główne wykresy
    ax_orig = fig.add_subplot(gs[0, :])
    ax_wiener = fig.add_subplot(gs[1, 0])
    ax_savgol = fig.add_subplot(gs[1, 1])
    ax_emd = fig.add_subplot(gs[1, 2])
    ax_comparison = fig.add_subplot(gs[2, :])
    ax_spectra = fig.add_subplot(gs[3, :])
    
    plt.subplots_adjust(bottom=0.15)
    
    # Suwaki
    ax_noise = plt.axes([0.1, 0.08, 0.2, 0.03])
    ax_wiener_len = plt.axes([0.35, 0.08, 0.2, 0.03])
    ax_savgol_win = plt.axes([0.6, 0.08, 0.2, 0.03])
    ax_emd_imf = plt.axes([0.85, 0.08, 0.1, 0.03])
    
    slider_noise = Slider(ax_noise, 'Poziom szumu', 0.1, 1.0, valinit=0.4)
    slider_wiener = Slider(ax_wiener_len, 'Wiener filter', 10, 100, valinit=30)
    slider_savgol = Slider(ax_savgol_win, 'S-G okno', 5, 51, valinit=15)
    slider_emd = Slider(ax_emd_imf, 'EMD IMFs', 1, 6, valinit=3)
    
    def wiener_denoise(signal_data, filter_param):
        """Uproszczona implementacja filtru Wienera"""
        # Filtr dolnoprzepustowy jako aproksymacja Wienera
        b, a = signal.butter(4, filter_param/500, 'low')  # fs=1000 zakładane
        return signal.filtfilt(b, a, signal_data)
    
    def update_all(val):
        # Czyszczenie wykresów
        for ax in [ax_orig, ax_wiener, ax_savgol, ax_emd, ax_comparison, ax_spectra]:
            ax.clear()
        
        # Parametry
        noise_level = slider_noise.val
        wiener_param = int(slider_wiener.val)
        savgol_window = int(slider_savgol.val)
        if savgol_window % 2 == 0:
            savgol_window += 1
        emd_imfs = int(slider_emd.val)
        
        # Dodanie kontrolowanego szumu
        np.random.seed(42)  # dla powtarzalności
        noise = np.random.randn(len(signal_data)) * noise_level
        noisy_data = signal_data + noise
        
        # Filtracja Wienera
        filtered_wiener = wiener_denoise(noisy_data, wiener_param)
        
        # Filtracja Savitzky-Golay
        poly_order = min(3, savgol_window-1)
        filtered_savgol = savgol_filter(noisy_data, savgol_window, poly_order)
        
        # Filtracja EMD
        emd = EMD()
        try:
            IMFs = emd.emd(noisy_data)
            num_imfs = min(emd_imfs, len(IMFs))
            filtered_emd = np.sum(IMFs[:num_imfs], axis=0)
        except:
            filtered_emd = noisy_data  # fallback
        
        # Wykresy
        t = np.arange(len(signal_data)) / 1000  # zakładając fs=1000
        
        ax_orig.plot(t, noisy_data, 'k-', alpha=0.7)
        ax_orig.set_title(f'Sygnał z pliku CSV (poziom szumu: {noise_level:.2f})')
        ax_orig.grid(True)
        
        ax_wiener.plot(t, filtered_wiener, 'b-', linewidth=2)
        ax_wiener.set_title(f'Filtr Wienera (param: {wiener_param})')
        ax_wiener.grid(True)
        
        ax_savgol.plot(t, filtered_savgol, 'g-', linewidth=2)
        ax_savgol.set_title(f'Savitzky-Golay (okno: {savgol_window})')
        ax_savgol.grid(True)
        
        ax_emd.plot(t, filtered_emd, 'r-', linewidth=2)
        ax_emd.set_title(f'EMD ({emd_imfs} IMFs)')
        ax_emd.grid(True)
        
        # Porównanie wszystkich metod
        ax_comparison.plot(t, noisy_data, 'k-', alpha=0.5, label='Zaszumiony')
        ax_comparison.plot(t, filtered_wiener, 'b-', label='Wiener', linewidth=2)
        ax_comparison.plot(t, filtered_savgol, 'g-', label='Savitzky-Golay', linewidth=2)
        ax_comparison.plot(t, filtered_emd, 'r-', label='EMD', linewidth=2)
        ax_comparison.set_title('Porównanie wszystkich metod')
        ax_comparison.legend()
        ax_comparison.grid(True)
        
        # Porównanie widm
        freqs = np.fft.fftfreq(len(signal_data), 1/1000)[:len(signal_data)//2]
        
        def get_spectrum(sig):
            return np.abs(np.fft.fft(sig))[:len(sig)//2]
        
        ax_spectra.plot(freqs, get_spectrum(noisy_data), 'k-', alpha=0.5, label='Zaszumiony')
        ax_spectra.plot(freqs, get_spectrum(filtered_wiener), 'b-', label='Wiener')
        ax_spectra.plot(freqs, get_spectrum(filtered_savgol), 'g-', label='S-G')
        ax_spectra.plot(freqs, get_spectrum(filtered_emd), 'r-', label='EMD')
        ax_spectra.set_title('Porównanie widm amplitudowych')
        ax_spectra.set_xlabel('Częstotliwość [Hz]')
        ax_spectra.set_ylabel('Amplituda')
        ax_spectra.legend()
        ax_spectra.grid(True)
        ax_spectra.set_xlim(0, 300)
        
        plt.draw()
    
    # Połączenie suwaków
    slider_noise.on_changed(update_all)
    slider_wiener.on_changed(update_all)
    slider_savgol.on_changed(update_all)
    slider_emd.on_changed(update_all)
    
    update_all(None)
    plt.show()



if __name__== "__main__":
    zadanie1()
    #zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
    zadanie6()
    zadanie7()
