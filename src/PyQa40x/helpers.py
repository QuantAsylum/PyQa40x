import numpy as np

def remove_dc(array: np.ndarray) -> np.ndarray:
    """
    Removes the DC component from an array by subtracting the mean of the array.

    Args:
        array (np.ndarray): The input array from which to remove the DC component.

    Returns:
        np.ndarray: The array with the DC component removed.
    """
    return array - np.mean(array)

def dbv_to_vpk(dbv: float) -> float:
    """
    Convert dBV to peak voltage.

    Parameters:
    dbv (float): Amplitude in dBV.

    Returns:
    float: Peak voltage.
    """
    return 10 ** (dbv / 20) * np.sqrt(2)


def dbfs_to_dbv(dbfs: float) -> float:
    """
    Convert dBFS to dBV.

    Parameters:
    dbfs (float): Amplitude in dBFS.

    Returns:
    float: Amplitude in dBV.
    """
    return dbfs - 2.98


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

def dbv_to_linear_pk(dbv: float) -> float:
    """
    Converts dBV to linear peak voltage.

    Args:
        dbv (float): Amplitude in dBV.

    Returns:
        float: Amplitude in linear peak voltage.
    """
    linear_rms = dBV_to_linear(np.float64(dbv))
    linear_peak = linear_rms * np.sqrt(2)
    return float(linear_peak)

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

# Functions to handle float arrays

def linear_array_to_dBV(values: np.ndarray) -> np.ndarray:
    """
    Convert an array of linear values to dBV.

    Parameters:
    - values: The array of linear values to convert (numpy ndarray).

    Returns:
    - The corresponding array of values in dBV (numpy ndarray).
    """
    dBV = 20 * np.log10(values)
    return dBV

def linear_array_to_dBu(values: np.ndarray) -> np.ndarray:
    """
    Convert an array of linear values to dBu (decibels relative to 0.775 volts).

    Parameters:
    - values: The array of linear values to convert (numpy ndarray).

    Returns:
    - The corresponding array of values in dBu (numpy ndarray).
    """
    ref_voltage = 0.775
    dBu = 20 * np.log10(values / ref_voltage)
    return dBu

def dBV_array_to_linear(dBV_values: np.ndarray) -> np.ndarray:
    """
    Convert an array of dBV values back to linear.

    Parameters:
    - dBV_values: The array of dBV values to convert (numpy ndarray).

    Returns:
    - The corresponding array of linear values (numpy ndarray).
    """
    linear_values = 10**(dBV_values / 20.0)
    return linear_values

def dBu_array_to_linear(dBu_values: np.ndarray) -> np.ndarray:
    """
    Convert an array of dBu values back to linear.

    Parameters:
    - dBu_values: The array of dBu values to convert (numpy ndarray).

    Returns:
    - The corresponding array of linear values (numpy ndarray).
    """
    ref_voltage = 0.775
    linear_values = 10**((dBu_values / 20.0) + np.log10(ref_voltage))
    return linear_values
