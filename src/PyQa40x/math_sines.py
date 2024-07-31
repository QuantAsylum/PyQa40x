import numpy as np
from scipy.fftpack import fft, fftfreq
from typing import Optional

from PyQa40x.math_energy import compute_rms_freq

def compute_thd_linear(signal_freq_lin: np.ndarray, frequencies: np.ndarray, fundamental: float, num_harmonics: int = 5, debug: bool = False) -> float:
    """
    Computes the Total Harmonic Distortion (THD) of a given signal spectrum as a linear value.

    Parameters:
    signal_freq_lin (np.ndarray): The input signal's linear magnitude spectrum.
    frequencies (np.ndarray): The frequency array corresponding to the signal spectrum.
    fundamental (float): The specified fundamental frequency.
    num_harmonics (int): The number of harmonics to include in the THD calculation.
    debug (bool): If True, print debug information.

    Returns:
    float: The THD of the signal as a linear value.
    """
    window_Hz_pm = 10.0  # Set the window for searching the actual fundamental and harmonics

    # Find the actual fundamental frequency within the specified window
    lower_bound = fundamental - window_Hz_pm
    upper_bound = fundamental + window_Hz_pm
    fundamental_indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
    if len(fundamental_indices) == 0:
        raise ValueError(f"No fundamental frequency found within the specified window ({lower_bound} Hz to {upper_bound} Hz).")
    
    fundamental_idx = fundamental_indices[np.argmax(signal_freq_lin[fundamental_indices])]
    fundamental_amplitude = signal_freq_lin[fundamental_idx]

    # Debugging: Show the peak amplitude in dB
    if debug:
        fundamental_amplitude_db = 20 * np.log10(fundamental_amplitude)
        print(f"Fundamental Frequency: {frequencies[fundamental_idx]:.2f} Hz (bin {fundamental_idx})")
        print(f"Fundamental Amplitude: {fundamental_amplitude:.6f} (Linear), {fundamental_amplitude_db:.2f} dB")

    # Calculate the sum of squares of the harmonic amplitudes
    harmonic_amplitudes_sq_sum = 0.0
    for n in range(2, num_harmonics + 1):
        harmonic_freq = n * fundamental
        lower_bound_harmonic = harmonic_freq - window_Hz_pm
        upper_bound_harmonic = harmonic_freq + window_Hz_pm
        
        harmonic_indices = np.where((frequencies >= lower_bound_harmonic) & (frequencies <= upper_bound_harmonic))[0]
        
        if len(harmonic_indices) == 0:
            if debug:
                print(f"No harmonic indices found within the specified window for {n}x harmonic.")
            continue

        harmonic_idx = harmonic_indices[np.argmax(signal_freq_lin[harmonic_indices])]
        harmonic_amplitude = signal_freq_lin[harmonic_idx] if harmonic_idx < len(signal_freq_lin) else 0.0
        harmonic_amplitudes_sq_sum += harmonic_amplitude ** 2

        # Debugging: Show the harmonic amplitude in dB and the bins being examined
        if debug:
            harmonic_amplitude_db = 20 * np.log10(harmonic_amplitude)
            print(f"{n}x Harmonic Frequency: {harmonic_freq:.2f} Hz (closest bin {harmonic_idx})")
            print(f"{n}x Harmonic Amplitude: {harmonic_amplitude:.6f} (Linear), {harmonic_amplitude_db:.2f} dB")
            
            # Additional debugging: Show the amplitudes of the bins around the harmonic
            for offset in [-2, -1, 0, 1, 2]:
                idx = harmonic_idx + offset
                if 0 <= idx < len(signal_freq_lin):
                    amplitude = signal_freq_lin[idx]
                    amplitude_db = 20 * np.log10(amplitude)
                    print(f"Bin {idx} Frequency: {frequencies[idx]:.2f} Hz, Amplitude: {amplitude:.6f} (Linear), {amplitude_db:.2f} dB")

    # Compute THD
    thd = np.sqrt(harmonic_amplitudes_sq_sum) / fundamental_amplitude

    # Debugging: Show THD computation details
    if debug:
        print(f"Sum of Squares of Harmonic Amplitudes: {harmonic_amplitudes_sq_sum:.6f}")
        print(f"THD: {thd:.6f} (Linear)")

    return thd

def compute_thdn_linear(signal_freq_lin: np.ndarray, frequencies: np.ndarray, fundamental: float, notch_octaves: float = 0.5, start_freq: float = 20.0, stop_freq: float = 20000.0, debug: bool = False) -> float:
    """
    Computes the Total Harmonic Distortion plus Noise (THDN) of a given signal spectrum as a linear value.

    Parameters:
    signal_freq_lin (np.ndarray): The input signal's linear magnitude spectrum.
    frequencies (np.ndarray): The frequency array corresponding to the signal spectrum.
    fundamental (float): The specified fundamental frequency.
    notch_octaves (float): The bandwidth of the notch filter in plus/minus octaves around the fundamental.
    start_freq (float): The start frequency for the THDN measurement.
    stop_freq (float): The stop frequency for the THDN measurement.
    debug (bool): If True, print debug information.

    Returns:
    float: The THDN of the signal as a linear value.
    """
    # Calculate notch filter bounds in Hz
    notch_lower_bound = fundamental / (2 ** notch_octaves)
    notch_upper_bound = fundamental * (2 ** notch_octaves)
    
    if debug:
        print(f"Notch Filter Bounds: {notch_lower_bound:.2f} Hz to {notch_upper_bound:.2f} Hz")

    # Calculate RMS of the fundamental within the notch
    fundamental_rms = compute_rms_freq(signal_freq_lin, frequencies, notch_lower_bound, notch_upper_bound)
    
    if debug:
        print(f"Fundamental RMS: {fundamental_rms:.6f}")

    # Calculate RMS of the signal outside the notch
    rms_below_notch = compute_rms_freq(signal_freq_lin, frequencies, start_freq, notch_lower_bound)
    rms_above_notch = compute_rms_freq(signal_freq_lin, frequencies, notch_upper_bound, stop_freq)
    noise_rms = np.sqrt(rms_below_notch ** 2 + rms_above_notch ** 2)

    if debug:
        print(f"RMS Below Notch: {rms_below_notch:.6f}")
        print(f"RMS Above Notch: {rms_above_notch:.6f}")
        print(f"Noise RMS: {noise_rms:.6f}")

    # Calculate THDN
    thdn = noise_rms / fundamental_rms

    if debug:
        print(f"THDN: {thdn:.6f} (Linear)")

    return thdn
