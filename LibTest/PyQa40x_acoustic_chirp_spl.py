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

mic_ref_level = 94 # Sensitivity of mics are usually specified at this level ()
mic_sense = -51.5  # SURE SM58
preamp_gain = 20   # QA472

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=6, max_output_level=18, fft_size=2**15)

# Dump key parameters
print(params)

wc = WaveChirp(params)

wave_dac = wc.gen_chirp_dbv(-20)

wave_adc, _ = analyzer.send_receive(wave_dac, wave_dac)

freq, fft, ir, window = wc.compute_fft_db(wave_adc.get_main_buffer())

tsp = SeriesPlotter()
tsp.add_time_series(wave_dac.get_main_buffer(), "dac")
tsp.add_time_series(wave_adc.get_main_buffer(), "adc")
tsp.add_time_series(ir, "IR", signal_right = window)
tsp.add_freq_series(freq, fft, "FR", logx=True)
tsp.add_freq_series(freq, fft + mic_ref_level + mic_sense - preamp_gain, "FR w/Mic", logx=True)
tsp.plot();

analyzer.cleanup()
