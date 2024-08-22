import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..', 'src')))

from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.helpers import linear_to_dBV

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=0, max_output_level=8, fft_size=2**16, window_type="flattop")

# Dump key parameters
print(params)

# Wave class for DAC output
wave_dac = WaveSine(params).gen_sine_dbv(1000, 0)

# Wave class for first ADC input and accumulated ADC input
wave_adc_first = WaveSine(params)
wave_adc_accum = WaveSine(params)

# We'll do 64 acquisitions
num_acquisitions = 8  # Number of acquisitions to average

# iterate
for i in range(num_acquisitions):
    # Send the DAC buffers to the hardware, and collect the left ADC buffer
    print(f"Acquisition {i} of {num_acquisitions}")
    wave_adc, _ = analyzer.send_receive(wave_dac, wave_dac)

    # Special case: Save the first aquisition
    if i == 0:
        wave_adc_first.set_buffer(wave_adc.get_buffer())
       
    # Average in the time domain. If the input and output are not always precisely time aligned
    # by the same amount, the averaging in the time domain will converge to no signal. But, 
    # since QA403 acquisitions are precisely aligned in time, they can be averaged and the 
    # signal (and distortions) will get stronger and the noise will get weaker
    wave_adc_accum.set_buffer(wave_adc_accum.get_buffer() + wave_adc.get_buffer())
    
# Average
wave_adc_accum.set_buffer(wave_adc_accum.get_buffer()/num_acquisitions)

# Convert time-domain signals to freq domain, and specify dBV
freqs = wave_adc_first.get_frequency_array()
amplitude_first_dbv = wave_adc_first.get_amplitude_array("dbv")
amplitude_accumulated_dbv = wave_adc_accum.get_amplitude_array("dbv")

# Measure noise from 5.1 to 5.9 kHz of the first time series acquisitions
rms_first_5_to_6k = wave_adc_first.compute_rms_freq(5100, 5900)
print(f"RMS of First from 5100 to 5900 Hz: {rms_first_5_to_6k:0.2f}")

# measure noise from 5.1 to 5.9 kHz of the accumulated time series acquisitions
rms_accum_5_to_6k = wave_adc_accum.compute_rms_freq(5100, 5900)
print(f"RMS of accum from 5100 to 5900 Hz: {rms_accum_5_to_6k:0.2f}")

# Calc the noise win
reduction = rms_first_5_to_6k - rms_accum_5_to_6k

# Plotting
tsp = SeriesPlotter(num_columns=2)
tsp.add_freq_series(freqs, amplitude_first_dbv, "First Acquisition", logx=True, xmin=500, xmax = 10000)
tsp.add_freq_series(freqs, amplitude_accumulated_dbv, f"Accumulated Acquisition ({num_acquisitions} acquisitions, {reduction:0.1f} dB reduction in noise floor)", logx=True, xmin=500, xmax=10000)
tsp.plot()

analyzer.cleanup()
