import numpy as np
from PyQa40x.analyzer_params import AnalyzerParams

class Wave:
    def __init__(self, params: AnalyzerParams):
        """
        Initializes the Wave class with an instance of AnalyzerParams.
        
        Args:
            params (AnalyzerParams): An instance of the AnalyzerParams class.
        """
        self.params = params
        self.buffer: np.ndarray = np.zeros(self.params.pre_buf + self.params.fft_size + self.params.post_buf)
        

    def set_buffer(self, buffer: np.ndarray):
        """
        Sets the waveform buffer.

        Args:
            buffer (np.ndarray): The waveform buffer to set.
        """
        if buffer.shape != self.buffer.shape:
            raise ValueError("Buffer shape does not match the expected shape")
        self.buffer = buffer

    def get_buffer(self) -> np.ndarray:
        """
        Returns the current waveform buffer, including pre-, main- and post-buffer.

        Returns:
            np.ndarray: The current waveform buffer.
        """
        return self.buffer

    def get_main_buffer(self) -> np.ndarray:
        """
        Returns the central portion of the waveform buffer that matches the fft_size.

        Returns:
            np.ndarray: The central fft_size portion of the waveform buffer.
        """
        start_idx = self.params.pre_buf
        end_idx = start_idx + self.params.fft_size
        return self.buffer[start_idx:end_idx]

    def compute_instantaneous_dbspl(self, dbSpl_at_0dbv: float, rms_slice_interval_ms: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the instantaneous dB SPL values from the main waveform buffer.

        Args:
            dbSpl_at_0dbv (float): dB SPL at 0 dBV, encapsulating mic sensitivity and preamp gain.
            rms_slice_interval_ms (float): Interval in milliseconds to compute RMS values.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the dB SPL values and the corresponding time values.
        """
        # Extract the main portion of the buffer
        signal = self.get_main_buffer()
        
        # Convert slice interval from milliseconds to seconds
        segment_duration = rms_slice_interval_ms / 1000.0
        # Calculate the number of samples per segment
        segment_samples = int(segment_duration * self.params.sample_rate)
    
        dbspl_values = []
        
        # Iterate over the signal in segments
        for start in range(0, len(signal), segment_samples):
            segment = signal[start:start + segment_samples]
            if len(segment) == 0:
                break
            
            # Compute RMS value for the segment
            rms_value = np.sqrt(np.mean(segment**2))
            
            # Convert RMS value to sound pressure level in Pascals
            db_spl = 20 * np.log10(rms_value / 1) + dbSpl_at_0dbv
            dbspl_values.append(db_spl)
    
        # Generate corresponding time values
        time_values = np.linspace(0, len(signal) / self.params.sample_rate, len(dbspl_values))
        
        return np.array(dbspl_values), time_values
    
    def remove_dc(self):
        """
        Removes the DC component from the waveform buffer.
        """        
        # Subtract the mean value from the entire buffer to remove DC component
        self.buffer -= np.mean(self.buffer)
