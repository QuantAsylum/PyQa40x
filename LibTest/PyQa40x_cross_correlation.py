import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..', 'src')))

from PyQa40x.cross_correlation import CrossCorrelation
from PyQa40x.fft_processor import FFTProcessor
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave import Wave
from PyQa40x.wave_sine import WaveSine
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.helpers import linear_to_dBV
from PyQa40x.math_sines import compute_thdn_linear
from PyQa40x.math_energy import compute_rms_freq


iterations = 100
samples = 2**16
sample_rate = 48000
window_type = 'flattop'

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=0, max_output_level=18, fft_size=2**14, window_type="flattop")

# Dump key parameters
print(params)

wave_dac = WaveSine(params)
#wave_dac = wave_dac.gen_sine_dbv(1000, 0)

fp = FFTProcessor(params)
cc = CrossCorrelation()

# iterate
for i in range(iterations):
    # Send the DAC buffers to the hardware, and collect the left ADC buffer
    print(f"Acquisition {i} of {iterations}")
    wave_adc_left, wave_adc_right = analyzer.send_receive(wave_dac, wave_dac)

    freq_x = wave_adc_left.get_frequency_array()
    fft_left_dbv = wave_adc_left.get_amplitude_array();
    fft_right_dbv = wave_adc_right.get_amplitude_array()
    
    adc_left_windowed = wave_adc_left.get_main_buffer() * params.window
    adc_right_windowed = wave_adc_right.get_main_buffer() * params.window
    
    fft_complex_left = np.fft.fft(adc_left_windowed)
    fft_complex_right = np.fft.fft(adc_right_windowed)
    
    cc_result = cc.do_correlation(fft_complex_left, fft_complex_right)
   

# Full FFT length
N_full = len(cc_result)

# Calculate magnitude in dBV with length scaling
# Here we include the division by N_full to scale for FFT length
fft_mag = (np.abs(cc_result) / (N_full/2)) / np.sqrt(2)  # Scaling by FFT length
fft_mag = fft_mag * params.ACF


# Generate the full frequency spectrum
freqs_full = np.fft.rfftfreq(N_full, d=1/sample_rate)


thdn = compute_thdn_linear(fft_mag, freqs_full, 1000)
thdn_db = 20 * np.log10(thdn)

rms_noise = compute_rms_freq(fft_mag, freqs_full, 20, 2000)
rms_noise = rms_noise * params.ECF
rms_noise_db = 20*np.log10(rms_noise)


fft_mag_dbv = 20 * np.log10(fft_mag)  # Converting to dBV

# Determine the number of positive frequencies including Nyquist
Npos = N_full // 2 + 1

# Slice to take only the positive frequencies and corresponding magnitudes
freqs = freqs_full[:Npos]
fft_mag_dbv = fft_mag_dbv[:Npos]

print("Frequency Range:", freqs[0], "to", freqs[-2])
 
tsp = SeriesPlotter(num_columns=2)
tsp.add_freq_series(freqs, fft_left_dbv, "Left Acquisition", logx=True, xmin= 20, xmax=20000)
tsp.add_freq_series(freqs, fft_right_dbv, "Right Acquisition", logx=True, xmin= 20, xmax=20000)
tsp.newrow()
tsp.add_freq_series(freqs, fft_mag_dbv, f"Cross Correlation: {iterations} iterations rms dB = {rms_noise_db:0.2f}", logx=True, xmin= 20, xmax=20000)
tsp.plot()    
    
# Average
#wave_adc_accum.set_buffer(wave_adc_accum.get_buffer()/num_acquisitions)

# Convert time-domain signals to freq domain, and specify dBV
#freqs = wave_adc_first.get_frequency_array()
# amplitude_first_dbv = wave_adc_first.get_amplitude_array("dbv")
# amplitude_accumulated_dbv = wave_adc_accum.get_amplitude_array("dbv")

# # Measure noise from 5.1 to 5.9 kHz of the first time series acquisitions
# rms_first_5_to_6k = wave_adc_first.compute_rms_freq(5100, 5900)
# print(f"RMS of First from 5100 to 5900 Hz: {rms_first_5_to_6k:0.2f}")

# # measure noise from 5.1 to 5.9 kHz of the accumulated time series acquisitions
# rms_accum_5_to_6k = wave_adc_accum.compute_rms_freq(5100, 5900)
# print(f"RMS of accum from 5100 to 5900 Hz: {rms_accum_5_to_6k:0.2f}")

# # Calc the noise win
# reduction = rms_first_5_to_6k - rms_accum_5_to_6k

# # Plotting
# tsp = SeriesPlotter(num_columns=2)
# tsp.add_freq_series(freqs, amplitude_first_dbv, "First Acquisition", logx=True, xmin=500, xmax = 10000)
# tsp.add_freq_series(freqs, amplitude_accumulated_dbv, f"Accumulated Acquisition ({num_acquisitions} acquisitions, {reduction:0.1f} dB reduction in noise floor)", logx=True, xmin=500, xmax=10000)
# tsp.plot()

analyzer.cleanup()
