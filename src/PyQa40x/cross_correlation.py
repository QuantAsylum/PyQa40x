import numpy as np
from typing import Optional

class Average:
    def __init__(self):
        self.waveform_sum: Optional[np.ndarray] = None
        self.count: int = 0

    def add_waveform(self, waveform: np.ndarray) -> None:
        """
        Add a new waveform to the current sum.
        
        :param waveform: A NumPy array representing the waveform to add.
        """
        if self.waveform_sum is None:
            self.waveform_sum = np.copy(waveform)
            self.count = 1
        else:
            self.waveform_sum += waveform
            self.count += 1

    def clear(self) -> None:
        """
        Clear the current waveform sum and reset the count.
        """
        self.waveform_sum = None
        self.count = 0

    def calc_average(self) -> np.ndarray:
        """
        Calculate the average waveform from the sum.

        :return: A NumPy array representing the average waveform.
        :raises ValueError: If no waveforms have been added.
        """
        if self.count > 0:
            result = self.waveform_sum / self.count
            return np.sqrt(result)
        
        raise ValueError("Average doesn't have any elements")


class CrossCorrelation:
    def __init__(self):
        self.avg = Average()

    def clear(self) -> None:
        """
        Clear the accumulated average data.
        """
        CrossCorrelation.avg.clear()

    def do_correlation(self, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        """
        Compute the element-wise product of a1 with the complex conjugate of a2.

        Parameters:
        - a1 (np.ndarray): First complex array.
        - a2 (np.ndarray): Second complex array.

        Returns:
        - np.ndarray: The element-wise product of a1 and the complex conjugate of a2.
        """
        # Ensure the input arrays are complex
        a1 = np.asarray(a1, dtype=np.complex128)
        a2 = np.asarray(a2, dtype=np.complex128)

        # Compute the complex conjugate of a2
        a2_conjugate = np.conj(a2)

        # Perform element-wise multiplication
        product = a1 * a2_conjugate
        
        self.avg.add_waveform(product)

        # Always return the current vector average.
        return self.avg.calc_average()



