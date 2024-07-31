import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_chirp import WaveChirp
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.math_general import *
import PyQa40x.math_chirp as mc

# Python code used to play a chirp from a speaker, and measure the energy decay curve.
# The QA403 output goes into a +20 dB power amp (QA462), and that goes into a speaker.
# An SM58 mic is placed in the middle of the room to capture the chirp. That chirp
# is then converted into an impulse response, and the energy of that IR is plotted.
# The time for the energy to decay 20 dB is noted on the graph

mic_ref_level_dbspl = 94 # Mic sense reference level
mic_sense_dbv_at_ref_level = -28.9  # Earthworks M23R
preamp_gain = 0   # QA472

dbSpl_at_0dbv = mic_ref_level_dbspl - mic_sense_dbv_at_ref_level - preamp_gain

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=6, max_output_level=18, fft_size=2**18)

# Dump key parameters
print(params)

wc = WaveChirp(params)

wave_dac = wc.gen_chirp_dbv(-5)

# Send the DAC buffers to the hardware, and collect the left ADC buffer
wave_adc, _ = analyzer.send_receive(wave_dac, wave_dac)

# Remove DC
wave_adc.remove_dc()

# Compute RMS every 10 mS
adc_dbspl, _ = wave_adc.compute_instantaneous_dbspl(dbSpl_at_0dbv, 10)

# Compute frequency response, applying a window around the impulse response
req, fft, ir, window = wc.compute_fft_db(wave_adc.get_main_buffer(), window_start_time=0.05, window_end_time=2, ramp_up_time = 0.02, ramp_down_time=0.5)

# We're looking for the 20 dB decay point around the IR
db_decay = 20

# Apply the window and compute RT
rt, edc_db, start_idx, end_idx = wc.compute_rt(ir * window, db_decay)

# Plot 
tsp = SeriesPlotter(num_columns=2)
tsp.add_time_series(wave_adc.get_main_buffer(), "ADC")
tsp.add_time_series(adc_dbspl, "ADC dBSPL")
tsp.newrow()
tsp.add_time_series(ir, "Room Impulse Response & Window", signal_right = window)
tsp.add_time_series(edc_db, f"Energy Decay Curve ({db_decay}dB = {rt:.2f} sec)", ymax=10, ymin=-70)
tsp.plot();

analyzer.cleanup()
