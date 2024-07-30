import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PyQa40x.math_sines import compute_thd_db
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.math_general import *
import PyQa40x.math_chirp as mc

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=6, max_output_level=18, fft_size=2**18)

# Dump key parameters
print(params)

ws = WaveSine(params)

wave_dac = ws.gen_sine_dbv(1000, 0).gen_sine_dbv(2000, -50)

# Send the DAC buffers to the hardware, and collect the left ADC buffer
wave_adc, _ = analyzer.send_receive(wave_dac.get_buffer(), wave_dac.get_buffer())
                                    
if isinstance(wave_adc, WaveSine):
    thd_db = wave_adc.compute_thd_db(1000)

freq, spectrum = analyzer.get_amplitude_array()

# Plot 
tsp = SeriesPlotter(num_columns=2)
tsp.add_time_series(wave_adc.get_main_buffer(), "ADC")
tsp.newrow()
tsp.add_freq_series(freq, spectrum, "Spectrum")
tsp.plot();

analyzer.cleanup()
