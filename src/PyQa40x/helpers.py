import numpy as np

def linear_to_dBV(value: np.float64) -> np.float64:
    """
    Convert a linear value to dBV.

    Parameters:
    - value: The linear value to convert (numpy float64).

    Returns:
    - The corresponding value in dBV (numpy float64).
    """
    dBV = 20 * np.log10(value)
    return dBV

def linear_to_dBu(value: np.float64) -> np.float64:
    """
    Convert a linear value to dBu (decibels relative to 0.775 volts).

    Parameters:
    - value: The linear value to convert (numpy float64).

    Returns:
    - The corresponding value in dBu (numpy float64).
    """
    ref_voltage = 0.775
    dBu = 20 * np.log10(value / ref_voltage)
    return dBu

def dBV_to_linear(dBV: np.float64) -> np.float64:
    """
    Convert dBV value back to linear.

    Parameters:
    - dBV: The dBV value to convert (numpy float64).

    Returns:
    - The corresponding linear value (numpy float64).
    """
    linear_value = 10**(dBV / 20.0)
    return linear_value

def dBu_to_linear(dBu: np.float64) -> np.float64:
    """
    Convert dBu value back to linear.

    Parameters:
    - dBu: The dBu value to convert (numpy float64).

    Returns:
    - The corresponding linear value (numpy float64).
    """
    ref_voltage = 0.775
    linear_value = 10**((dBu / 20.0) + np.log10(ref_voltage))
    return linear_value
