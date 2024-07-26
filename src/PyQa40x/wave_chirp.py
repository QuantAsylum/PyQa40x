import numpy as np
from PyQa40x.analyzer_params import AnalyzerParams
from PyQa40x.math_general import normalize_spectrum
from PyQa40x.wave import Wave
from PyQa40x.helpers import dbv_to_vpk
import PyQa40x.math_chirp as mc
import PyQa40x.math_general as mg

class WaveChirp(Wave):
    def __init__(self, params: AnalyzerParams):
        # Initialize the parent class with the AnalyzerParams instance
        super().__init__(params)
        self.chirp_buf = None
        self.inv_filter = None
        self.ref_chirp_buf = None
        self.ref_inv_filter = None
        self.ref_freq = None
        self.ref_fft = None
        
    def gen_chirp_dbv(self, dbv: float, chirp_start_freq: float = 20, chirp_stop_freq: float = 20000, chirp_width: float = 0.6) -> 'WaveChirp':
        """
        Generates a chirp signal with a specified amplitude in dBV.

        Parameters:
        dbv (float): Amplitude of the chirp signal in dBV.
        chirp_start_freq (float): Start frequency of the chirp. Default is 20 Hz.
        chirp_stop_freq (float): End frequency of the chirp. Default is 20,000 Hz.
        chirp_width (float): Percentage of the buffer length used for the chirp signal. Default is 0.6.

        Returns:
        'WaveChirp': The WaveChirp instance with the generated chirp signal.
        """
    
        # Convert dBV to peak voltage
        vpk: float = dbv_to_vpk(dbv)
    
        # Generate the chirp buffer and inverse filter
        # self.fft_size: Total length of the FFT buffer
        # self.sample_rate: Sampling frequency
        self.chirp_buf, self.inv_filter = mc.chirp_vp(self.params.fft_size, self.params.sample_rate, vpk, chirp_start_freq, chirp_stop_freq, chirp_width)

        # Grab a reference version for 0 dBV = 1.41Vp
        self.ref_chirp_buf, self.ref_inv_filter = mc.chirp_vp(self.params.fft_size, self.params.sample_rate, np.sqrt(2), chirp_start_freq, chirp_stop_freq, chirp_width)
        self.ref_freq, self.ref_fft, _, _ = mc.normalize_and_compute_fft(self.ref_chirp_buf, self.ref_inv_filter, self.params.sample_rate)
        #print(f"Value @ 1k {np.interp(1000, self.ref_freq, self.ref_fft)} dB")

        # Concatenate pre-buffer, chirp buffer, and post-buffer to form the final buffer
        # self.pre_buf: Length of the pre-buffer (filled with zeros)
        # self.post_buf: Length of the post-buffer (filled with zeros)
        self.buffer = np.concatenate((np.zeros(self.params.pre_buf), self.chirp_buf, np.zeros(self.params.post_buf)))
    
        return self

    def get_buffer_and_invfilter(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the buffer and the inverse filter.

        Returns:
            tuple[np.ndarray, np.ndarray]: The buffer and the inverse filter.
        """
        return self.buffer, self.inv_filter

    def compute_fft_db_OLD(self, chirp: np.ndarray = None, inverse_filter: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the impulse response and the FFT of a DUT chirp signal, normalizes the FFT,
        and converts it to dBV. Uses the instance's chirp_buf and inv_filter by default.

        Args:
            dut_chirp (np.ndarray, optional): The chirp signal from the device under test (DUT).
                                                Defaults to the instance's chirp_buf.
            inverse_filter (np.ndarray, optional): The inverse filter to be convolved with the chirp.
                                                    Defaults to the instance's inv_filter.
            target_sample_rate (float, optional): The sample rate used for the chirp signal.
                                                    Defaults to the instance's sample_rate.

        Returns:
            tuple: A tuple containing:
                - freq (np.ndarray): The frequency bins for the FFT.
                - fft_dut_db (np.ndarray): The normalized FFT of the DUT chirp in dB.
                - ir_dut (np.ndarray): The impulse response of the DUT chirp.
        """
        if chirp is None:
            chirp = self.chirp_buf
        if inverse_filter is None:
            inverse_filter = self.inv_filter
        
        # Get the needed correction for the reference
        _, correction_db = mg.normalize_spectrum(self.ref_freq, self.ref_fft, 1000);
        
        freq, fft, ir = mc.normalize_and_compute_fft(chirp, self.ref_inv_filter, self.params.sample_rate);
        #print(f"Value @ 1k {np.interp(1000, freq, fft)} dB")
        
        fft = fft + correction_db
        #print(f"Value @ 1k {np.interp(1000, freq, fft)} dB")
        
        return freq, fft, ir
    
    def compute_fft_db(self, chirp: np.ndarray = None, inverse_filter: np.ndarray = None,
                       window_start_time: float = 0.001, window_end_time: float = 0.005,
                       ramp_up_time: float = 0.001, ramp_down_time: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the impulse response and the FFT of a DUT chirp signal, normalizes the FFT,
        converts it to dBV, and generates a window for the impulse response.

        Args:
            chirp (np.ndarray, optional): The chirp signal from the device under test (DUT).
                                          Defaults to the instance's chirp_buf.
            inverse_filter (np.ndarray, optional): The inverse filter to be convolved with the chirp.
                                                   Defaults to the instance's inv_filter.
            window_start_time (float, optional): The time before the IR peak for the window to start (in seconds).
                                                 Defaults to 0.001 seconds.
            window_end_time (float, optional): The time after the IR peak for the window to stop (in seconds).
                                               Defaults to 0.005 seconds.
            ramp_up_time (float, optional): The ramp-up time of the window (in seconds).
                                            Defaults to 0.001 seconds.
            ramp_down_time (float, optional): The ramp-down time of the window (in seconds).
                                              Defaults to 0.001 seconds.

        Returns:
            tuple: A tuple containing:
                - freq (np.ndarray): The frequency bins for the FFT.
                - fft_dut_db (np.ndarray): The normalized FFT of the DUT chirp in dB.
                - ir_dut (np.ndarray): The impulse response of the DUT chirp.
                - window (np.ndarray): The generated window function.
        """
        if chirp is None:
            chirp = self.chirp_buf
        if inverse_filter is None:
            inverse_filter = self.inv_filter
    
        # Get the needed correction for the reference
        _, correction_db = mg.normalize_spectrum(self.ref_freq, self.ref_fft, 1000)
    
        freq, fft, ir, window = mc.normalize_and_compute_fft(
            chirp, inverse_filter, self.params.sample_rate, 
            window_start_time, window_end_time, ramp_up_time, ramp_down_time
        )
    
        fft = fft + correction_db
    
        return freq, fft, ir, window
    
    def compute_rt(self, ir, decay_db):
        """
        Calculates the reverberation time (RT) from the impulse response (IR) of a system.

        Args:
            ir (np.ndarray): The impulse response of the system.
            sample_rate (float): The sample rate of the impulse response.
            decay_db (float): The decay level in decibels for which to calculate the RT (e.g., 60 for RT60).

        Returns:
            tuple: A tuple containing:
                - rt (float): The calculated reverberation time.
                - edc_db (np.ndarray): The energy decay curve in decibels.
                - start_idx (int): The start index for the decay calculation.
                - end_idx (int): The end index for the decay calculation.
        """
        # Square the impulse response to get energy
        squared_ir = ir ** 2
    
        # Compute the energy decay curve (EDC) by cumulative summation of squared IR in reverse
        edc = np.cumsum(squared_ir[::-1])[::-1]
    
        # Convert the EDC to decibels, normalizing by the maximum value
        edc_db = 10 * np.log10(edc / np.max(edc))
    
        # Find the start index where the EDC drops below -5 dB
        start_idx = np.where(edc_db <= -5)[0][0]
    
        # Find the end index where the EDC drops below (-5 - decay_db) dB
        end_idx = np.where(edc_db <= -5 - decay_db)[0][0]
    
        # Perform a linear fit to the EDC in the range from start_idx to end_idx
        slope, intercept = np.polyfit(np.arange(start_idx, end_idx) / self.params.sample_rate, edc_db[start_idx:end_idx], 1)
    
        # Calculate the reverberation time (RT)
        rt = -(decay_db + 5) / slope
    
        return rt, edc_db, start_idx, end_idx
