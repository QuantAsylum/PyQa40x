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

# Python code used to sweep the frequency response of a speaker.

mic_ref_level_dbspl = 94            # Mic sense reference level
mic_sense_dbv_at_ref_level = -28.9  # Earthworks M23R
preamp_gain = 0                     # QA472

dbSpl_at_0dbv = mic_ref_level_dbspl - mic_sense_dbv_at_ref_level - preamp_gain

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=6, max_output_level=18, fft_size=2**17)

# Dump key parameters
print(params)

wc = WaveChirp(params)

# Gen a -20 dBV chirp. This will be gained up 20 dB by the QA472 and driven into a speaker
wave_dac = wc.gen_chirp_dbv(-20)

wave_adc, _ = analyzer.send_receive(wave_dac, wave_dac)

freq, fft, ir, window = wc.compute_fft_db(wave_adc.get_main_buffer(), window_start_time=0.010, window_end_time=1)

tsp = SeriesPlotter()
tsp.add_time_series(wave_dac.get_main_buffer(), "dac")
tsp.add_time_series(wave_adc.get_main_buffer(), "adc")
tsp.add_time_series(ir, "IR", signal_right = window)
tsp.add_freq_series(freq, fft, "FR (dB)", logx=True)
tsp.add_freq_series(freq, fft + dbSpl_at_0dbv, "FR dB SPL", logx=True)
tsp.plot();

analyzer.cleanup()
