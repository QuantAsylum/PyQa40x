import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.helpers import linear_to_dBV

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=6, max_output_level=18, fft_size=2**14)

# Dump key parameters
print(params)

ws = WaveSine(params)

wave_dac = ws.gen_sine_dbv(1000, 0).gen_sine_dbv(2000, -50).gen_sine_dbv(3000, -50)

# Send the DAC buffers to the hardware, and collect the left ADC buffer
wave_adc, _ = analyzer.send_receive(wave_dac, wave_dac)
                                    
thd_db = wave_adc.compute_thd_db(1000)
thd_pct = wave_adc.compute_thd_pct(1000)

# compute total energy of entire spectrum
total_energy_dbv = wave_adc.compute_rms_freq(20, 20000)
twoH_plus_threeH_energy = wave_adc.compute_rms_freq(1500, 3500)
print(f"Total Energy: {total_energy_dbv:.2f} dBV")
print(f"2H + 3H Energy: {twoH_plus_threeH_energy:.2f} dBV")

freqs = wave_adc.get_frequency_array()
amps_dbv = wave_adc.get_amplitude_array()

# Plot 
tsp = SeriesPlotter(num_columns=2)
tsp.add_time_series(wave_adc.get_main_buffer(), "ADC")
tsp.newrow()
tsp.add_freq_series(freqs, amps_dbv, f"Spectrum (THD = {thd_db:.2f} dB / {thd_pct:.2f}%   RMS (20-20k)={total_energy_dbv:0.2f}dBV  RMS (1.5k to 3.5k)={twoH_plus_threeH_energy:0.2f} dBV)")
tsp.plot();

analyzer.cleanup()
