import numpy as np
from PyQa40x.analyzer_params import AnalyzerParams
from PyQa40x.fft_processor import FFTProcessor
from PyQa40x.helpers import *
import PyQa40x.math_sines as ms
import PyQa40x.math_energy as me


class Wave:
    def __init__(self, params: AnalyzerParams):
        """
        Initializes the Wave class with an instance of AnalyzerParams.
        
        Args:
            params (AnalyzerParams): An instance of the AnalyzerParams class.
        """
        self.params = params
        self.buffer: np.ndarray = np.zeros(self.params.pre_buf + self.params.fft_size + self.params.post_buf)

        self.fft_plot: FFTProcessor | None = None
        self.fft_energy: FFTProcessor | None = None
        self.fft_plot_signal: np.ndarray | None = None
        self.fft_energy_signal: np.ndarray | None = None

        self.amplitude_unit: str = "dbv"  # Lowercase. Use dbv or dbu or volts
        self.energy_unit: str = "dbv"     # Lowercase. Use dbv or dbu or volts
        self.distortion_unit: str = "db"  # Lowercase. Use db or pct
        

    def set_buffer(self, buffer: np.ndarray):
        """
        Sets the waveform buffer.

        Args:
            buffer (np.ndarray): The waveform buffer to set.
        """
        if buffer.shape != self.buffer.shape:
            raise ValueError("Buffer shape does not match the expected shape")

        self.fft_plot = None
        self.fft_energy = None
        self.fft_plot_signal = None
        self.fft_energy_signal = None
        self.buffer = buffer
        self.fft_plot = None
        self.fft_energy = None

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

    def compute_fft(self):
        self.fft_plot = FFTProcessor(self.params)
        self.fft_energy = FFTProcessor(self.params)

        self.fft_plot_signal = self.fft_plot.fft_forward(self.get_main_buffer()).apply_acf().get_result();
        self.fft_energy_signal = self.fft_energy.fft_forward(self.get_main_buffer()).apply_ecf().get_result()
        
    def compute_fft_if_needed(self):
        if self.fft_plot is None or self.fft_energy is None:
            self.compute_fft()
            

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

    def compute_thd_db(self, fundamental: float, window_pmHz: float = 100.0, num_harmonics: int = 5) -> float:
        """
        Computes the Total Harmonic Distortion (THD) of the sine wave in dB.

        Parameters:
        fundamental (Optional[float]): The specified fundamental frequency. If None, uses the instance's frequency.
        window (float): The frequency window around the fundamental to search for the actual fundamental.
        num_harmonics (int): The number of harmonics to include in the THD calculation.

        Returns:
        float: The THD of the signal in dB.
        """

        self.compute_fft_if_needed()
        return ms.compute_thd_db(self.fft_plot_signal, self.get_frequency_array(), fundamental, window_pmHz, num_harmonics)

    def compute_thd_pct(self, fundamental: float = None, window: float = 100.0, num_harmonics: int = 5) -> float:
        """
        Computes the Total Harmonic Distortion (THD) of the sine wave in percentage.

        Parameters:
        fundamental (Optional[float]): The specified fundamental frequency. If None, uses the instance's frequency.
        window (float): The frequency window around the fundamental to search for the actual fundamental.
        num_harmonics (int): The number of harmonics to include in the THD calculation.

        Returns:
        float: The THD of the signal as a percentage.
        """
        self.compute_fft_if_needed()
        return ms.compute_thd_pct(self.fft_plot_signal, self.get_frequency_array(), fundamental, window, num_harmonics)

    def compute_rms_freq(self, start_freq: float, stop_freq: float, unit: str = None, debug: bool = False) -> float:
        """
        Computes the RMS value of the sine wave over a specified frequency range.

        Parameters:
        start_freq (float): The start frequency of the range over which to compute the RMS value.
        stop_freq (float): The end frequency of the range over which to compute the RMS value.
        unit (str): Unit for the amplitude ('dbv' or 'dbu'). Defaults to the instance's default unit.
        debug (bool): If True, print debug information.

        Returns:
        float: The RMS value of the signal in the specified frequency range.
        """
        self.compute_fft_if_needed()
        rms_value = me.compute_rms_freq(self.fft_energy_signal, self.get_frequency_array(), start_freq, stop_freq, debug)
        
        # Convert RMS value to the specified unit
        if unit is None:
            unit = self.amplitude_unit

        if unit == "dbv":
            rms_value_db = linear_to_dBV(rms_value)
        elif unit == "dbu":
            rms_value_db = linear_to_dBu(rms_value)
        else:
            raise ValueError(f"Unknown amplitude units: {unit}")

        if debug:
            print(f"RMS Value ({unit}): {rms_value_db:.2f} dB")

        return rms_value_db
    
    def remove_dc(self):
        """
        Removes the DC component from the waveform buffer.
        """        
        # Subtract the mean value from the entire buffer to remove DC component
        self.buffer -= np.mean(self.buffer)
        
    def get_frequency_array(self) -> np.ndarray:
        """
        Returns an array of frequencies given the user-specified parameters.

        Returns:
            np.ndarray: Array of frequencies.
        """
        self.compute_fft_if_needed()
        return self.fft_plot.get_frequencies()
    
    def get_amplitude_array(self, amplitude_unit: str | None = None) -> np.ndarray:
        """
        Returns an array of amplitudes based on the specified units.

        Args:
            amplitude_unit (str | None): Unit for amplitude measurements, default is None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing arrays of left and right channel amplitudes.
        """
        self.compute_fft_if_needed()
        
        if amplitude_unit is None:
            amplitude_unit = self.amplitude_unit
            
        if amplitude_unit == "dbv":
            fft_dB = linear_array_to_dBV(self.fft_plot_signal)
        elif amplitude_unit == "dbu":
            fft_dB = linear_array_to_dBu(self.fft_plot_signal)
        else:
            raise ValueError(f"Unknown amplitude units: {amplitude_unit}") 
            
        return fft_dB
