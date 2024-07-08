PyQa40x allows you to run Python code to communicate with the QA40x hardware. No intermediate program is needed, as the PyQa40x lib understands how to use the calibration data stored inside the QA40x hardware.

To get started, you can install the library in the Jupyter Lab environment by typing the following into a Juptyer Lab cell and the running the command:

```
!pip install git+https://github.com/QuantAsylum/PyQa40x.git
```

After the installation has finished, you can paste the following into another cell:

```
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
```

With your QA40x connected, you can run the code above, and you should be greated with the following output:

![image](https://github.com/QuantAsylum/PyQa40x/assets/27789827/cadab538-e595-4655-a564-f4f4f226f8df)

In the sample above, the left channel was connected in single-ended loopback, and the right channel had shorting blocks applied. In the code, you can see we created a waveform with two tones: a 0 dBV tone at 1 kHz, and a -50 dBV tone at 2 kHz. The amplitudes are correctly plotted. 

Next, an energy calc was done from 20 to 20 kHz and the returned number matched expectations. 

## What's Next

More functionality will be moved into the analyzer class, including the ability to calculate THD, THD+N, amplitude, frequency, etc. 

A plot class is included. The aim here is to get the basic plots down to a line of code. More work is needed there.

Chirp will be added to facilitate frequency response and phase plots. 

Stronger hinting and comments will hopefully enabled ChatGPT or other LLMs to become useful helpers for generating Jupyter notebooks with test. 

