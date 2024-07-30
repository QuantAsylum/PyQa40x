import numpy as np
from scipy.fftpack import fft, fftfreq
from typing import Optional

def compute_thd_db(signal: np.ndarray, sample_rate: float, fundamental: float, window: float = 100.0, num_harmonics: int = 5) -> float:
    """
    Computes the Total Harmonic Distortion (THD) of a given signal in dB.

    Parameters:
    signal (np.ndarray): The input signal.
    sample_rate (float): The sample rate of the signal.
    fundamental (float): The specified fundamental frequency.
    window (float): The frequency window around the fundamental to search for the actual fundamental.
    num_harmonics (int): The number of harmonics to include in the THD calculation.

    Returns:
    float: The THD of the signal in dB.
    """
    # Perform FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)

    # Get the magnitude of the FFT
    yf_magnitude = np.abs(yf) / N

    # Find the actual fundamental frequency within the specified window
    lower_bound = fundamental - window
    upper_bound = fundamental + window
    fundamental_idx = np.argmax(yf_magnitude[(xf >= lower_bound) & (xf <= upper_bound)])
    fundamental_idx += np.where(xf >= lower_bound)[0][0]  # Adjust index based on the slice

    fundamental_amplitude = yf_magnitude[fundamental_idx]

    # Calculate the sum of squares of the harmonic amplitudes
    harmonic_amplitudes_sq_sum = 0.0
    for n in range(2, num_harmonics + 1):
        harmonic_idx = n * fundamental_idx
        if harmonic_idx < len(yf_magnitude):
            harmonic_amplitudes_sq_sum += yf_magnitude[harmonic_idx] ** 2

    # Compute THD
    thd = np.sqrt(harmonic_amplitudes_sq_sum) / fundamental_amplitude
    thd_db = 20 * np.log10(thd)

    return thd_db

def compute_thd_pct(signal: np.ndarray, sample_rate: float, fundamental: float, window: float = 100.0, num_harmonics: int = 5) -> float:
    """
    Computes the Total Harmonic Distortion (THD) of a given signal in percentage.

    Parameters:
    signal (np.ndarray): The input signal.
    sample_rate (float): The sample rate of the signal.
    fundamental (float): The specified fundamental frequency.
    window (float): The frequency window around the fundamental to search for the actual fundamental.
    num_harmonics (int): The number of harmonics to include in the THD calculation.

    Returns:
    float: The THD of the signal as a percentage.
    """
    thd_db = compute_thd_db(signal, sample_rate, fundamental, window, num_harmonics)
    thd_percent = 10 ** (thd_db / 20) * 100
    return thd_percent
