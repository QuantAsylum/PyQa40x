import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.fft_processor import FFTProcessor
from PyQa40x.sig_proc import SigProc
from PyQa40x.helpers import *

# Create an Analyzer instance
analyzer = Analyzer()

# Initialize the analyzer with desired parameters
params = analyzer.init(sample_rate=48000, max_input_level=0, max_output_level=18, fft_size = 4096, window_type = 'hamming')

# Dump key parameters
print(params)

# Create a WaveSine instance, passing the Analyzer instance
wave_sine = WaveSine(analyzer)

# Generate a 0 dBV 1 kHz tone and a -50 dBV 2 kHz tone using chaining
left_dac_data = wave_sine.gen_sine_dbv(1000, 0).gen_sine_dbv(2000, -50).get_buffer()

# Use the same waveform for the right channel
right_dac_data = left_dac_data

# Send and receive the signals from the QA40x hardware
left_adc_data, right_adc_data = analyzer.send_receive(left_dac_data, right_dac_data)

# the buffers we send/receive have three regions:-pre-buffer, post-buffer and main buffer
# Here, we take just the main buffer (which matches the fft_size we specified above)
left_adc_data = left_adc_data.get_main_buffer();
right_adc_data = right_adc_data.get_main_buffer();

fft_plot = FFTProcessor(params)
fft_rms = FFTProcessor(params)

# Do a forward fft on the received data, apply the amplitude correction factor for the given
# window, convert to dbV, and display
fft_plot_left, fft_plot_right = fft_plot.fft_forward(left_adc_data, right_adc_data).apply_acf().to_dbv().get_result()
fft_rms_left, fft_rms_right = fft_rms.fft_forward(left_adc_data, right_adc_data).apply_ecf().get_result()

# Generate an array of freqs to display as the x-axis on the graph
fft_freqs = fft_plot.get_frequencies()

# Plot the first N samples of the received left and right channels
num_samples_to_plot = 500
plt.figure(figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.plot(left_adc_data[:num_samples_to_plot])
plt.title(f'Left ADC - First {num_samples_to_plot} Samples')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 2)
plt.plot(right_adc_data[:num_samples_to_plot])
plt.title(f'Right ADC - First {num_samples_to_plot} Samples')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 3)
plt.plot(fft_freqs, fft_plot_left)
plt.title('Left ADC Spectrum')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (dBV)')

plt.subplot(2, 2, 4)
plt.plot(fft_freqs, fft_plot_right)
plt.title('Right ADC Spectrum')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (dBV)')

plt.tight_layout()
plt.show()

sig = SigProc(params)
rms_left_20k = sig.compute_rms(fft_rms_left, 20, 20000)
rms_left_20k_dbv = 20 * np.log10(rms_left_20k)

rms_right_20k = sig.compute_rms(fft_rms_right, 20, 20000)
rms_right_20k_dbv = 20 * np.log10(rms_right_20k)

print(f'Energy 20k Left: {rms_left_20k_dbv:.2f} dBV  Right: {rms_right_20k_dbv:.2f} dBV')

# Clean up resources
analyzer.cleanup()
