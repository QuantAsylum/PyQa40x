import numpy as np
from scipy.signal import windows

# THIS CLASS SHOULD BE REMOVED AS IT'S NOT BEING USED

class SigProc:
    def __init__(self, params):
        self.params = params

    def compute_rms(self, fft_data: np.ndarray, start_freq: float, stop_freq: float) -> float:
        """
        Compute the RMS value of linear FFT data across a given frequency range.

        Args:
            fft_data (np.ndarray): Linear FFT data.
            start_freq (float): Start frequency for RMS computation.
            stop_freq (float): Stop frequency for RMS computation.

        Returns:
            float: RMS value across the specified frequency range.
        """
        # Compute the frequency bins
        freqs = np.fft.rfftfreq(len(fft_data) * 2 - 1, 1 / self.params.sample_rate)
        
        # Find the indices corresponding to the start and stop frequencies
        start_idx = np.searchsorted(freqs, start_freq)
        stop_idx = np.searchsorted(freqs, stop_freq)

        #print(f"rms start index: {start_idx}")
        #print(f"rms stop index: {stop_idx}")
        
        # Extract the relevant portion of the FFT data
        relevant_fft_data = fft_data[start_idx:stop_idx]

        # Compute the RMS value
        rms_value = np.sqrt(np.sum(relevant_fft_data**2))
        
        return rms_value

    def to_dBV(self, linear_value: float) -> float:
        """
        Convert linear value to dBV.

        Args:
            linear_value (float): Linear value.

        Returns:
            float: Value in dBV.
        """
        return 20 * np.log10(linear_value)

    def to_dBu(self, linear_value: float) -> float:
        """
        Convert linear value to dBu.

        Args:
            linear_value (float): Linear value.

        Returns:
            float: Value in dBu.
        """
        return 20 * np.log10(linear_value) + 2.2
