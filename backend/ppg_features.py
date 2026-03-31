import numpy as np
from scipy.fft import fft


def extract_ppg_features(signal):
    """
    Extract 12 features from a PPG segment for signal quality classification.
    
    Features:
    - mean, std, skewness, kurtosis (statistical moments)
    - signal_range, peak_to_peak (amplitude features)
    - zero_crossings, rms (signal variation)
    - spectral_energy, dominant_freq (frequency domain)
    - peaks_count, valleys_count (local extrema)
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 10:
        return np.zeros(12)
    
    # 1. Basic statistics
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-7:
        std = 1e-7
    
    diff = arr - mean
    
    # 2. Higher moments
    skewness = np.mean(np.power(diff, 3)) / (np.power(std, 3) + 1e-7)
    kurtosis = np.mean(np.power(diff, 4)) / (np.power(std, 4) + 1e-7)
    
    # 3. Amplitude features
    signal_range = np.max(arr) - np.min(arr)
    peak_to_peak = signal_range
    
    # 4. Signal variation
    zero_crossings = np.sum(np.abs(np.diff(np.sign(diff)))) // 2
    rms = np.sqrt(np.mean(np.square(arr)))
    
    # 5. Frequency domain features (FFT)
    n = len(arr)
    if n > 0:
        fft_vals = np.abs(fft(arr))[:n // 2]
        spectral_energy = np.sum(np.square(fft_vals))
        # Dominant frequency index (normalized)
        if len(fft_vals) > 1:
            dominant_freq = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
            dominant_freq = dominant_freq / n  # Normalize
        else:
            dominant_freq = 0
    else:
        spectral_energy = 0
        dominant_freq = 0
    
    # 6. Local extrema (peaks and valleys)
    peaks = 0
    valleys = 0
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks += 1
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            valleys += 1
    
    return np.array([
        mean, std, skewness, kurtosis,
        signal_range, zero_crossings,
        rms, peak_to_peak,
        spectral_energy, dominant_freq,
        peaks, valleys
    ], dtype=float)