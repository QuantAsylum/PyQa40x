import numpy as np

from PyQa40x.helpers import *

class FFTProcessor:
    def __init__(self, params):
        self.params = params
        self.time_series = None
        self.windowed_time_series = None
        self.fft_data= None

    def fft_forward(self, signal: np.ndarray) -> 'FFTProcessor':
        """
        Compute the forward FFT of the given signals.

        Args:
            left_signal (np.ndarray): Input signal for the left channel.
            right_signal (np.ndarray, optional): Input signal for the right channel. Defaults to None.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        self.time_series = signal
        self.windowed_time_series = signal * self.params.window
        fft_result = np.fft.rfft(self.windowed_time_series)
        self.fft_data = (np.abs(fft_result) / (self.params.fft_size / 2)) / np.sqrt(2)
        
        return self

    def apply_acf(self) -> 'FFTProcessor':
        """
        Apply the Amplitude Correction Factor (ACF) to the FFT data.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data is not None:
            self.fft_data *= self.params.ACF

        return self

    def apply_ecf(self) -> 'FFTProcessor':
        """
        Apply the Energy Correction Factor (ECF) to the FFT data.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data is not None:
            self.fft_data *= self.params.ECF

        return self

    def to_dbv(self) -> 'FFTProcessor':
        """
        Convert FFT data to dBV (decibels relative to 1 volt).

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data is not None:
            self.fft_data = linear_array_to_dBV(self.fft_data)
        return self

    def to_dbu(self) -> 'FFTProcessor':
        """
        Convert FFT data to dBu (decibels relative to 0.775 volts).

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """

        if self.fft_data is not None:
            self.fft_data = linear_array_to_dBu(self.fft_data)
        return self

    def get_result(self) -> tuple[np.ndarray]:
        """
        Get the current FFT data for both channels at any stage of processing.

        Returns:
            tuple: The current FFT data for left and right channels.
        """
        return self.fft_data

    def get_frequencies(self) -> np.ndarray:
        """
        Get the frequency bins corresponding to the FFT.

        Returns:
            np.ndarray: Frequency bins.
        """
        return np.fft.rfftfreq(self.params.fft_size, 1 / self.params.sample_rate)
