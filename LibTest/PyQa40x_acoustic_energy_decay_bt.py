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

# Python code used to play a chirp from a speaker or bluetooth, and measure the energy decay curve.
# The QA403 output goes into a +20 dB power amp (QA462), and that goes into a speaker.
# An SM58 mic is placed in the middle of the room to capture the chirp. That chirp
# is then converted into an impulse response, and the energy of that IR is plotted.
# The time for the energy to decay 20 dB is noted on the graph

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
#bluetooth_device = "SineAudio DS6301+"
bluetooth_device = ""

# Define chirp amplitude to be used in dBV. This will be used if
# bluetooth isn't being used OR if test_loopback is specified
chirp_amp_dbv = -10

# Define chirp amplitude to used in dBFS. This will be used if
# blueotooth is used
chirp_amp_dbfs = -20

# Mute left or right as desired
mute_left_dac = False
mute_right_dac = False

mic_ref_level_dbspl = 94            # Mic sense reference level dBspl
mic_sense_dbv_at_ref_level = -28.9  # Earthworks M23R
preamp_gain = 0                     # QA472

dbspl_at_0dbv = mic_ref_level_dbspl - mic_sense_dbv_at_ref_level - preamp_gain

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
params = analyzer.init(sample_rate=sample_rate, max_input_level=6, max_output_level=18, fft_size=2**18, bt_device_name=bluetooth_device)

# Dump key parameters
print(params)

wave_chirp = WaveChirp(params)

if test_loopback or bluetooth_device == "":
    print(f"db_spl_at_0dbv: {dbspl_at_0dbv:0.1f}")
    if mute_left_dac is False:
        wave_dac_left = wave_chirp.gen_chirp_dbv(chirp_amp_dbv)
    if mute_right_dac is False:    
        wave_dac_right = wave_chirp.gen_chirp_dbv(chirp_amp_dbv)
else:
    if mute_left_dac is False:
        wave_dac_left = wave_chirp.gen_chirp_dbfs(chirp_amp_dbfs)
    if mute_right_dac is False:    
        wave_dac_right = wave_chirp.gen_chirp_dbfs(chirp_amp_dbfs)  

# Send the DAC buffers to the hardware, and collect the left ADC buffer
wave_adc_left, _ = analyzer.send_receive(wave_dac_left, wave_dac_right)

# Remove DC
wave_adc_left.remove_dc()

# Compute peak RMS every 10 mS
adc_dbspl, _ = wave_adc_left.compute_instantaneous_dbspl(dbspl_at_0dbv, 10)

# Compute frequency response, applying a window around the impulse response
req, fft, ir, window = wave_chirp.compute_fft_db(wave_adc_left.get_main_buffer(), window_start_time=0.05, window_end_time=2, ramp_up_time = 0.02, ramp_down_time=0.5, apply_window=True)

# We're looking for the 20 dB decay point around the IR
db_decay = 20

# Apply the window and compute RT
rt, edc_db, start_idx, end_idx = wave_chirp.compute_rt(ir, db_decay)

# Plot 
tsp = SeriesPlotter(num_columns=2)
tsp.add_time_series(wave_adc_left.get_main_buffer(), "ADC")
tsp.add_time_series(adc_dbspl, "ADC dBSPL")
tsp.newrow()
tsp.add_time_series(ir, "Room Impulse Response & Window", signal_right = window)
tsp.add_time_series(edc_db, f"Energy Decay Curve ({db_decay}dB = {rt:.2f} sec)", ymax=10, ymin=-70)
tsp.plot();

analyzer.cleanup()
