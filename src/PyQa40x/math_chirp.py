import numpy as np

from typing import Tuple
from scipy.signal import fftconvolve
from scipy.fft import fft

from PyQa40x.math_windows import *

def chirp_vp(total_buffer_length: int, fs: float, amplitude_vpk: float, f1: float = 20, f2: float = 20000, pct: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a chirp signal and its inverse filter.

    Parameters:
    total_buffer_length (int): Total length of the buffer.
    fs (float): Sampling frequency.
    amplitude_vpk (float): Amplitude of the chirp signal in peak volts.
    f1 (float): Start frequency of the chirp. Default is 20 Hz.
    f2 (float): End frequency of the chirp. Default is 20,000 Hz.
    pct (float): Percentage (0 to 1.0) of the buffer length used for the chirp signal. Default is 0.6.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the padded chirp signal and its inverse filter.
    """
    
    # Calculate the length of the chirp in samples
    chirp_length_samples = int(total_buffer_length * pct)
    
    # Duration of the chirp in seconds
    T = chirp_length_samples / fs
    
    # Time array for the chirp duration
    t = np.arange(0, T, 1 / fs)
    
    # Chirp rate (logarithmic)
    R = np.log(f2 / f1)
    
    # Generate the chirp signal
    chirp_signal = amplitude_vpk * np.sin((2 * np.pi * f1 * T / R) * (np.exp(t * R / T) - 1))
    
    # Calculate the start and end indices of the window
    window_start = 0
    window_end = chirp_length_samples
    ramp_length = chirp_length_samples // 10  # Fixed ramp-up and ramp-down length
    
    # Apply window function
    window = generate_window(len(chirp_signal), window_start, window_end, ramp_length, ramp_length)
    chirp_signal = chirp_signal * window[:len(chirp_signal)]
       
    # Calculate the required padding length
    padding = total_buffer_length - len(chirp_signal)
    
    # Pad the chirp signal with zeros to fit the total buffer length
    padded_chirp = np.pad(chirp_signal, (0, padding), 'constant')
    
    # Time array for the padded chirp
    padded_t = np.arange(0, len(padded_chirp)) / fs
    
    # Scaling factor for the inverse filter
    k = np.exp(padded_t * R / T)
    
    # Generate the inverse filter by reversing and scaling the chirp signal
    inverse_filter = padded_chirp[::-1] / k
    
    return padded_chirp, inverse_filter

def normalize_and_compute_fft_OLD(chirp: np.ndarray, inverse_filter: np.ndarray, target_sample_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the impulse response and the FFT of a DUT chirp signal, normalizes the FFT,
    and converts it to dB.

    Args:
        chirp (np.ndarray): The chirp signal from the device under test (DUT).
        inverse_filter (np.ndarray): The inverse filter to be convolved with the chirp.
        target_sample_rate (float): The sample rate used for the chirp signal.

    Returns:
        tuple: A tuple containing:
            - freq (np.ndarray): The frequency bins for the FFT.
            - fft_dut_dbv (np.ndarray): The normalized FFT of the DUT chirp in dB.
            - ir_dut (np.ndarray): The impulse response of the DUT chirp.
    """
    
    # Compute impulse response by convolving DUT chirp with inverse filter
    ir = fftconvolve(chirp, inverse_filter, mode='same')

    # Compute FFT of the impulse response
    fft_dut = fft(ir)
    fft_dut = fft_dut[:len(fft_dut) // 2]  # Take the positive frequency components

    # Compute frequency bins
    freq = np.fft.fftfreq(len(chirp), 1 / target_sample_rate)
    freq = freq[:len(freq) // 2]

    # Normalize FFT by the length of the FFT
    fft_dut_normalized = np.abs(fft_dut) / len(fft_dut)

    # Convert to dBV
    fft_dut_db = 20 * np.log10(fft_dut_normalized)

    return freq, fft_dut_db, ir

def normalize_and_compute_fft(chirp: np.ndarray, inverse_filter: np.ndarray, target_sample_rate: float, 
                              window_start_time: float = 0.005, window_end_time: float = 0.01, 
                              ramp_up_time: float = 0.0001, ramp_down_time: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the impulse response and the FFT of a DUT chirp signal, normalizes the FFT,
    converts it to dB, and generates a window for the impulse response.

    Args:
        chirp (np.ndarray): The chirp signal from the device under test (DUT).
        inverse_filter (np.ndarray): The inverse filter to be convolved with the chirp.
        target_sample_rate (float): The sample rate used for the chirp signal.
        window_start_time (float): The time before the IR peak for the window to start (in seconds).
        window_end_time (float): The time after the IR peak for the window to stop (in seconds).
        ramp_up_time (float): The ramp-up time of the window (in seconds).
        ramp_down_time (float): The ramp-down time of the window (in seconds).

    Returns:
        tuple: A tuple containing:
            - freq (np.ndarray): The frequency bins for the FFT.
            - fft_dut_dbv (np.ndarray): The normalized FFT of the DUT chirp in dB.
            - ir_dut (np.ndarray): The impulse response of the DUT chirp.
            - window (np.ndarray): The generated window function.
    """
    
    # Compute impulse response by convolving DUT chirp with inverse filter
    ir = fftconvolve(chirp, inverse_filter, mode='same')
    
    # Compute FFT of the impulse response
    fft_dut = fft(ir)
    fft_dut = fft_dut[:len(fft_dut) // 2]  # Take the positive frequency components

    # Compute frequency bins
    freq = np.fft.fftfreq(len(chirp), 1 / target_sample_rate)
    freq = freq[:len(freq) // 2]

    # Normalize FFT by the length of the FFT
    fft_dut_normalized = np.abs(fft_dut) / len(fft_dut)

    # Convert to dBV
    fft_dut_db = 20 * np.log10(fft_dut_normalized)
    
    # Determine window parameters in samples
    window_start = int(window_start_time * target_sample_rate)
    window_end = int(window_end_time * target_sample_rate)
    ramp_up = int(ramp_up_time * target_sample_rate)
    ramp_down = int(ramp_down_time * target_sample_rate)
    
    # Generate window
    peak_index = np.argmax(np.abs(ir))
    buffer_size = len(ir)
    window = generate_window(buffer_size, peak_index - window_start, peak_index + window_end, ramp_up, ramp_down)
    
    return freq, fft_dut_db, ir, window