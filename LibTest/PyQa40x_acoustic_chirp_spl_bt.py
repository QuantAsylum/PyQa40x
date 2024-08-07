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

#
# User Options
#

# Set sample rate
sample_rate = 48000

# If we enable this with the QA40x in loopback, then the dBSPL will be reported as 
# as db_spl_at_0dbv. Bluetooth will not be used if doing a loopback test
test_loopback = False

# Set to true to see a list of available bluetooth devices. This will list the
# devices for the sample rate specified above and then exit
bluetooth_list_devices = False

# Specify the name of your bluetooth device here. The chirp will be mirrored to that
# device.
bluetooth_device = "SineAudio DS6301+"
#bluetooth_device = ""

# Define chirp amplitude to be used in dBV. This will be used if
# bluetooth isn't being used OR if test_loopback is specified
chirp_amp_dbv = 0

# Define chirp amplitude to used in dBFS. This will be used if
# blueotooth is used
chirp_amp_dbfs = -10

# Mute left or right as desired
mute_left_dac = False
mute_right_dac = False

# Setup external gains and sensitivities
mic_ref_level_dbspl = 94            # Mic sense reference level
mic_sense_dbv_at_ref_level = -28.9  # Earthworks M23R
preamp_gain = 0                     # QA472

# Define a quantity for this combination of mic and pre-amp: What is the dBSPL
# measured when 0 dBV is seen at inputs
db_spl_at_0dbv = mic_ref_level_dbspl - mic_sense_dbv_at_ref_level - preamp_gain

#
# Make measurement
#

# if loopback test, override any bluetooth devices
if (test_loopback):
    bluetooth_device = ""

if bluetooth_list_devices == True:
    Analyzer.list_audio_devices_by_sample_rate(sample_rate)
    sys.exit()

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=sample_rate, max_input_level=6, max_output_level=18, fft_size=2**17, bt_device_name=bluetooth_device, pre_buf=16384)

# Dump key parameters
print(params)

# Create a chirp instance. Buffers are all zero'd (no sound)
wave_dac_left = WaveChirp(params)
wave_dac_right = WaveChirp(params)

if test_loopback or bluetooth_device == "":
    print(f"db_spl_at_0dbv: {db_spl_at_0dbv:0.1f}")
    if mute_left_dac is False:
        wave_dac_left = wave_dac_left.gen_chirp_dbv(chirp_amp_dbv)
    if mute_right_dac is False:    
        wave_dac_right = wave_dac_right.gen_chirp_dbv(chirp_amp_dbv)
else:
    if mute_left_dac is False:
        wave_dac_left = wave_dac_left.gen_chirp_dbfs(chirp_amp_dbfs)
    if mute_right_dac is False:    
        wave_dac_right = wave_dac_right.gen_chirp_dbfs(chirp_amp_dbfs)    
    
# Send chirp to left channel.
wave_adc_left, wave_adc_right = analyzer.send_receive(wave_dac_left, wave_dac_right)

freq, fft, ir, window = wave_dac_left.compute_fft_db(wave_adc_left.get_main_buffer(), window_start_time=0.005, window_end_time=0.25, ramp_down_time=0.1)

tsp = SeriesPlotter()
tsp.add_time_series(wave_dac_left.get_buffer(), "DAC Left")
tsp.add_time_series(wave_dac_right.get_buffer(), "DAC Right")
tsp.newrow()
tsp.add_time_series(wave_adc_left.get_buffer(), "ADC Left")
tsp.add_time_series(wave_adc_right.get_buffer(), "ADC Right")
tsp.newrow()
tsp.add_time_series(ir, "IR", signal_right = window)
tsp.add_freq_series(freq, fft, "FR (dB)", logx=True)
tsp.add_freq_series(freq, fft + db_spl_at_0dbv, "FR dB SPL", logx=True, units = "dBSPL", ymax=120, ymin=90)
tsp.plot();

analyzer.cleanup()