import numpy as np


def _hankel_matrix(x: np.ndarray, m: int) -> np.ndarray:
    n = len(x)
    if m >= n:
        raise ValueError("Hankel window length must be smaller than signal length")
    cols = n - m + 1
    return np.stack([x[i : i + m] for i in range(cols)], axis=1)


def music_spectrum(signal: np.ndarray, num_sources: int = 1, n_fft: int = 256) -> np.ndarray:
    """Compute a simple 1D MUSIC pseudospectrum from a signal.

    This uses a Hankel data matrix to form a covariance estimate.
    """
    x = np.asarray(signal).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=0)
    if np.iscomplexobj(x):
        x = x
    else:
        x = x.astype(np.float64)

    m = min(64, max(8, len(x) // 4))
    X = _hankel_matrix(x, m)
    R = (X @ X.conj().T) / X.shape[1]

    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)
    noise_vecs = eigvecs[:, idx[:-num_sources]]

    freqs = np.linspace(0, 1, n_fft, endpoint=False)
    spectrum = np.zeros_like(freqs, dtype=np.float64)
    for i, f in enumerate(freqs):
        steering = np.exp(-2j * np.pi * f * np.arange(m))
        denom = np.linalg.norm(noise_vecs.conj().T @ steering) ** 2
        spectrum[i] = 1.0 / max(denom, 1e-12)
    return spectrum
