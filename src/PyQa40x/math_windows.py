import numpy as np
import matplotlib.pyplot as plt

def generate_window(buffer_size: int, window_start: int, window_end: int, ramp_up: int, ramp_down: int) -> np.ndarray:
    """
    Generates a window function and places it in a buffer. This window function should be used for IR. For more common
    windowing, such as flattop, boxcar, etc on acquired time series, use standard np

    Args:
        buffer_size (int): The size of the buffer.
        window_start (int): The start index of the window in the buffer.
        window_end (int): The end index of the window in the buffer.
        ramp_up (int): The length of the ramp-up (Hann window) at the start of the window.
        ramp_down (int): The length of the ramp-down (Hann window) at the end of the window.

    Returns:
        np.ndarray: The buffer containing the window function.
        
    Raises:
        ValueError: If window start or end indices are out of buffer bounds.
    """
    if window_start < 0 or window_end > buffer_size:
        raise ValueError(
            f"[math_windows.generate_window] Error: Window start index ({window_start}) or end index ({window_end}) is out of buffer bounds (0 to {buffer_size})."
        )

    # Initialize the buffer with a very small value
    small_value = 1e-10
    buffer = np.full(buffer_size, small_value)
    
    # Define the length of the flat top region
    flat_top_length = max(0, window_end - window_start - ramp_up - ramp_down)
    
    # Adjust ramp lengths if they exceed the window bounds
    ramp_up = min(ramp_up, window_end - window_start)
    ramp_down = min(ramp_down, window_end - window_start - ramp_up)
    
    # Generate the Hann window for the ramp-up
    if ramp_up > 0:
        ramp_up_window = np.hanning(2 * ramp_up)[:ramp_up]
    else:
        ramp_up_window = np.array([])  # Empty array if no ramp-up
    
    # Generate the Hann window for the ramp-down
    if ramp_down > 0:
        ramp_down_window = np.hanning(2 * ramp_down)[ramp_down:]
    else:
        ramp_down_window = np.array([])  # Empty array if no ramp-down
    
    # Create the complete window function by concatenating ramp-up, flat top, and ramp-down
    window_function = np.concatenate([
        ramp_up_window,
        np.ones(flat_top_length),  # Flat top region
        ramp_down_window
    ])
    
    # Ensure the window function fits within the buffer size
    if window_start + len(window_function) > buffer_size:
        window_function = window_function[:buffer_size - window_start]
    
    # Place the window function in the buffer at the specified location
    buffer[window_start:window_start + len(window_function)] = window_function
    
    return buffer
