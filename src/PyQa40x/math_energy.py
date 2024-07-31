import numpy as np
from typing import Optional

def compute_rms_freq(signal_freq_lin: np.ndarray, frequencies: np.ndarray, freq_start: float, freq_end: float, debug: bool = False) -> float:
    """
    Computes the RMS value of a given signal spectrum over a specified frequency range.

    Parameters:
    signal_freq_lin (np.ndarray): The input signal's linear magnitude spectrum.
    frequencies (np.ndarray): The frequency array corresponding to the signal spectrum.
    freq_start (float): The start frequency of the range over which to compute the RMS value.
    freq_end (float): The end frequency of the range over which to compute the RMS value.
    debug (bool): If True, print debug information.

    Returns:
    float: The RMS value of the signal in the specified frequency range.
    """
    # Find the indices within the specified frequency range
    freq_indices = np.where((frequencies >= freq_start) & (frequencies <= freq_end))[0]

    # Compute the sum of squares of the amplitudes in the specified range
    sum_squares = np.sum(signal_freq_lin[freq_indices] ** 2)

    # Compute the RMS value
    rms = np.sqrt(sum_squares)

    # Debugging: Show computation details
    if debug:
        print(f"Frequency Range: {freq_start:.2f} Hz to {freq_end:.2f} Hz")
        print(f"Indices in Range: {freq_indices}")
        print(f"Sum of Squares: {sum_squares:.6f}")
        print(f"RMS Value: {rms:.6f}")

    return rms


