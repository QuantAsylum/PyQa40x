import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import Analyzer

class Wave:
    def __init__(self, analyzer: 'Analyzer'):
        """
        Initializes the Wave class with an instance of Analyzer.
        
        Args:
            analyzer (Analyzer): An instance of the Analyzer class.
        """
        self.analyzer = analyzer
        self.sample_rate: int = analyzer.params.sample_rate
        self.pre_buf: int = analyzer.params.pre_buf
        self.post_buf: int = analyzer.params.post_buf
        self.fft_size: int = analyzer.params.fft_size
        self.buffer: np.ndarray = np.zeros(self.pre_buf + self.fft_size + self.post_buf)

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
        start_idx = self.pre_buf
        end_idx = start_idx + self.fft_size
        return self.buffer[start_idx:end_idx]

class WaveSine(Wave):
    def gen_sine_dbv(self, frequency: float, dbv: float, snap_freq: bool = True) -> 'WaveSine':
        """
        Generates a sine wave of a given frequency and amplitude in dBV and adds it to the buffer.

        Args:
            frequency (float): Frequency of the sine wave in Hz.
            dbv (float): Amplitude of the sine wave in dBV.
            snap_freq (bool, optional): Whether to center the frequency in the FFT bin. Defaults to True.

        Returns:
            WaveSine: The instance of the WaveSine class to allow method chaining.
        """
        amplitude: float = self.dbv_to_linear_pk(dbv)
        return self._add_sine_wave(frequency, amplitude, snap_freq)

    def gen_sine_dbu(self, frequency: float, dbu: float, snap_freq: bool = True) -> 'WaveSine':
        """
        Generates a sine wave of a given frequency and amplitude in dBu and adds it to the buffer.

        Args:
            frequency (float): Frequency of the sine wave in Hz.
            dbu (float): Amplitude of the sine wave in dBu.
            snap_freq (bool, optional): Whether to center the frequency in the FFT bin. Defaults to True.

        Returns:
            WaveSine: The instance of the WaveSine class to allow method chaining.
        """
        dbv: float = dbu - 2.2  # Convert dBu to dBV
        amplitude: float = self.dbv_to_linear_pk(dbv)
        return self._add_sine_wave(frequency, amplitude, snap_freq)

    def gen_sine_vpk(self, frequency: float, vpk: float, snap_freq: bool = True) -> 'WaveSine':
        """
        Generates a sine wave of a given frequency and amplitude in peak volts and adds it to the buffer.

        Args:
            frequency (float): Frequency of the sine wave in Hz.
            vpk (float): Amplitude of the sine wave in peak volts.
            snap_freq (bool, optional): Whether to center the frequency in the FFT bin. Defaults to True.

        Returns:
            WaveSine: The instance of the WaveSine class to allow method chaining.
        """
        amplitude: float = vpk
        return self._add_sine_wave(frequency, amplitude, snap_freq)

    def _add_sine_wave(self, frequency: float, amplitude: float, snap_freq: bool) -> 'WaveSine':
        """
        Internal method to add a sine wave to the buffer.

        Args:
            frequency (float): Frequency of the sine wave in Hz.
            amplitude (float): Amplitude of the sine wave.
            snap_freq (bool): Whether to center the frequency in the FFT bin.

        Returns:
            WaveSine: The instance of the WaveSine class to allow method chaining.
        """
        num_samples: int = len(self.buffer)
        t: np.ndarray = np.arange(num_samples) / self.sample_rate

        if snap_freq:
            bin_resolution = self.sample_rate / self.fft_size
            frequency = round(frequency / bin_resolution) * bin_resolution
            print(f"_add_sine_wave: adjusted freq to {frequency}")

        sine_wave: np.ndarray = amplitude * np.sin(2 * np.pi * frequency * t)
        self.buffer += sine_wave

        return self

    @staticmethod
    def dbv_to_linear_pk(dbv: float) -> float:
        """
        Converts dBV to linear peak voltage.

        Args:
            dbv (float): Amplitude in dBV.

        Returns:
            float: Amplitude in linear peak voltage.
        """
        linear_rms: float = 10**(dbv / 20)
        linear_peak: float = linear_rms * np.sqrt(2)
        return linear_peak
