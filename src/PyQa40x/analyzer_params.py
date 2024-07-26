import scipy.signal # pip install scipy
import numpy as np

class AnalyzerParams:
    def __init__(self, sample_rate: int = 192000, max_input_level: int = 0, max_output_level: int = 18, 
                 pre_buf: int = 2048, post_buf: int = 2048, fft_size: int = 16384, window_type='boxcar'):
        """
        Initializes the AnalyzerParams class with default or specified parameters.

        Args:
            sample_rate (int): Sample rate for the hardware. Valid values are 48000, 96000, 192000.
            max_input_level (int): Maximum input level in dBV. Valid values are 0, 6, 12, 18, 24, 30, 36, 42.
            max_output_level (int): Maximum output level in dBV. Valid values are 18, 8, -2, -12.
            pre_buf (int): Size of the pre-buffer.
            post_buf (int): Size of the post-buffer.
            fft_size (int): Size of the FFT.
            window_type (str): Type of window function to apply to the signal before FFT.
        """
        self.sample_rate: int = sample_rate
        self.max_input_level: int = max_input_level
        self.max_output_level: int = max_output_level
        self.pre_buf: int = pre_buf
        self.post_buf: int = post_buf
        self.fft_size: int = fft_size

        self.window_type = window_type
        self.window = scipy.signal.get_window(self.window_type, self.fft_size)
        mean_w = np.mean(self.window)
        self.ACF = 1 / mean_w
        rms_w = np.sqrt(np.mean(self.window ** 2))
        self.ECF = 1 / rms_w

    def __str__(self) -> str:
        """
        Returns a string representation of the AnalyzerParams instance.

        Returns:
            str: String representation of the parameters formatted as a table.
        """
        params = [
            ("Sample Rate", f"{self.sample_rate} Hz"),
            ("Max Input Level", f"{self.max_input_level} dBV"),
            ("Max Output Level", f"{self.max_output_level} dBV"),
            ("Pre Buffer", f"{self.pre_buf}"),
            ("Post Buffer", f"{self.post_buf}"),
            ("Buffer Size", f"{self.fft_size}"),
            ("Duration", f"{self.fft_size/self.sample_rate:0.2f} sec"),
            ("Window Type", f"{self.window_type}")
        ]
        
        col_widths = [max(len(item[i]) for item in params) for i in range(2)]
        col_widths[1] += 5  # Add padding

        table = "===== ACQUISITION PARAMETERS =====\n"
        rows = len(params)
        cols = 3
        
        for row in range(0, rows, cols):
            row_str = ""
            for col in range(cols):
                if row + col < rows:
                    name, value = params[row + col]
                    row_str += f"{name.ljust(col_widths[0])} : {value.ljust(col_widths[1])}   "
            table += row_str.strip() + "\n"
        
        return table