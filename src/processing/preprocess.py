import numpy as np
from scipy.signal import butter, filtfilt, detrend as _detrend, savgol_filter


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to zero mean and unit variance."""
    mean = np.mean(signal)
    std = np.std(signal) + 1e-12
    return (signal - mean) / std


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend."""
    return _detrend(signal, type="linear")


def smooth_signal(signal: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    if window >= len(signal):
        return signal
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def preprocess_signal(
    signal: np.ndarray,
    fs: float = None,
    low: float = None,
    high: float = None,
) -> np.ndarray:
    """Efficient preprocessing: detrend -> smooth -> optional bandpass -> normalize."""
    x = np.asarray(signal).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=0)
    x = detrend_signal(x)
    x = smooth_signal(x)
    if fs and low and high:
        x = bandpass_filter(x, fs, low, high)
    x = normalize_signal(x)
    return x


def to_real_imag(signal: np.ndarray) -> np.ndarray:
    """Convert complex signal to 2-channel real/imag array."""
    if np.iscomplexobj(signal):
        return np.stack([signal.real, signal.imag], axis=0)
    return np.expand_dims(signal, axis=0)
