import numpy as np
from PyQa40x.analyzer_params import AnalyzerParams
from PyQa40x.wave import Wave
from PyQa40x.math_sines import *
from PyQa40x.helpers import *

class WaveSine(Wave):
    def __init__(self, params: AnalyzerParams):
        # Initialize the parent class with the AnalyzerParams instance
        super().__init__(params)
        
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
        amplitude: float = dbv_to_linear_pk(dbv)
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
        t: np.ndarray = np.arange(num_samples) / self.params.sample_rate

        if snap_freq:
            bin_resolution = self.params.sample_rate / self.params.fft_size
            frequency = round(frequency / bin_resolution) * bin_resolution
            print(f"_add_sine_wave: adjusted freq to {frequency}")

        sine_wave: np.ndarray = amplitude * np.sin(2 * np.pi * frequency * t)
        self.buffer += sine_wave

        return self
    
 


