import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.series_plotter import SeriesPlotter
from PyQa40x.helpers import linear_to_dBV

# Setup mic constants below, and then using a mic calibrator, verify the plotted fft peak
# matches the expected amplitude. For the test below, an M23R was used with a QA472 at 0 dB
# of gain. A 94 dBSPL mic calibrator was used. The 1 kHz peak in the FFT measured 94.12 dBSPL.
# Test was repeated at 114 dBSPL from the calibrator and also 10 dB from the pre-amp to 
# verify the results were again expected.

mic_ref_level_dbspl = 94            # Mic sense reference level
mic_sense_dbv_at_ref_level = -28.9  # Earthworks M23R
preamp_gain = 10                    # QA472

dbSpl_at_0dbv = mic_ref_level_dbspl - mic_sense_dbv_at_ref_level - preamp_gain

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=0, max_output_level=8, fft_size=2**14, window_type="flattop")

# Dump key parameters
print(params)

# By default, generates silence
ws = WaveSine(params)

# Send the DAC buffers to the hardware, and collect the left ADC buffer
wave_adc, _ = analyzer.send_receive(ws, ws)

# Get amplitude array (defaults to dBV) and freq array
adc_freq = wave_adc.get_frequency_array()
adc_amp_dbv = wave_adc.get_amplitude_array()

# Plot 
tsp = SeriesPlotter(num_columns=2)
tsp.add_freq_series(adc_freq, adc_amp_dbv + dbSpl_at_0dbv, "dBSPL", units="dBSPL")
tsp.plot();

analyzer.cleanup()
