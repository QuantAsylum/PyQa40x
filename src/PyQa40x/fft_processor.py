import numpy as np

class FFTProcessor:
    def __init__(self, params):
        self.params = params
        self.fft_data_left = None
        self.fft_data_right = None

    def fft_forward(self, left_signal: np.ndarray, right_signal: np.ndarray = None) -> 'FFTProcessor':
        """
        Compute the forward FFT of the given signals.

        Args:
            left_signal (np.ndarray): Input signal for the left channel.
            right_signal (np.ndarray, optional): Input signal for the right channel. Defaults to None.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        left_windowed = left_signal * self.params.window
        left_fft_result = np.fft.rfft(left_windowed)
        self.fft_data_left = (np.abs(left_fft_result) / (self.params.fft_size / 2)) / np.sqrt(2)

        if right_signal is not None:
            right_windowed = right_signal * self.params.window
            right_fft_result = np.fft.rfft(right_windowed)
            self.fft_data_right = (np.abs(right_fft_result) / (self.params.fft_size / 2)) / np.sqrt(2)
        else:
            self.fft_data_right = None

        return self

    def apply_acf(self) -> 'FFTProcessor':
        """
        Apply the Amplitude Correction Factor (ACF) to the FFT data.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data_left is not None:
            self.fft_data_left *= self.params.ACF
        if self.fft_data_right is not None:
            self.fft_data_right *= self.params.ACF
        return self

    def apply_ecf(self) -> 'FFTProcessor':
        """
        Apply the Energy Correction Factor (ECF) to the FFT data.

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data_left is not None:
            self.fft_data_left *= self.params.ECF
        if self.fft_data_right is not None:
            self.fft_data_right *= self.params.ECF
        return self

    def to_dbv(self) -> 'FFTProcessor':
        """
        Convert FFT data to dBV (decibels relative to 1 volt).

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        if self.fft_data_left is not None:
            self.fft_data_left = 20 * np.log10(self.fft_data_left)
        if self.fft_data_right is not None:
            self.fft_data_right = 20 * np.log10(self.fft_data_right)
        return self

    def to_dbu(self) -> 'FFTProcessor':
        """
        Convert FFT data to dBu (decibels relative to 0.775 volts).

        Returns:
            FFTProcessor: The instance itself to allow method chaining.
        """
        # First, convert to dBV
        self.to_dbv()

        dbv_to_dbu_conversion = 2.21  # dBV to dBu conversion factor
        if self.fft_data_left is not None:
            self.fft_data_left += dbv_to_dbu_conversion
        if self.fft_data_right is not None:
            self.fft_data_right += dbv_to_dbu_conversion
        return self

    def get_result(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current FFT data for both channels at any stage of processing.

        Returns:
            tuple: The current FFT data for left and right channels.
        """
        return self.fft_data_left, self.fft_data_right

    def get_frequencies(self) -> np.ndarray:
        """
        Get the frequency bins corresponding to the FFT.

        Returns:
            np.ndarray: Frequency bins.
        """
        return np.fft.rfftfreq(self.params.fft_size, 1 / self.params.sample_rate)
