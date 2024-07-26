import numpy as np
from typing import Tuple

def normalize_spectrum(freq: np.ndarray, magnitude_db: np.ndarray, norm_freq: float) -> Tuple[np.ndarray, float]:
    """
    Normalize the spectrum so that the amplitude at the normalization frequency is 0 dB.

    Parameters:
    freq (np.ndarray): Array of frequencies.
    magnitude_db (np.ndarray): Array of magnitudes in dB corresponding to the frequencies.
    norm_freq (float): The frequency at which the amplitude should be normalized to 0 dB.

    Returns:
    Tuple[np.ndarray, float]: 
        - Normalized magnitude array.
        - Normalization value that must be added to bring the amplitude at the specified frequency to 0 dB.
    """
    
    # Interpolate to find the magnitude at the normalization frequency
    norm_value = np.interp(norm_freq, freq, magnitude_db)
    
    # Calculate the normalization value (the value to be added to bring the magnitude at norm_freq to 0 dB)
    normalization_value = -norm_value
    
    # Normalize the magnitude array by adding the normalization value
    normalized_magnitude_db = magnitude_db + normalization_value
    
    return normalized_magnitude_db, normalization_value