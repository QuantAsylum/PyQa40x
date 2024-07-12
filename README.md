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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PyQa40x.analyzer import Analyzer
from PyQa40x.wave_sine import WaveSine
from PyQa40x.fft_processor import FFTProcessor
from PyQa40x.sig_proc import SigProc
from PyQa40x.helpers import *
from PyQa40x.series_plotter import SeriesPlotter

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

# Use the same waveform for the right channel, just invert
right_dac_data = -left_dac_data

print("Starting acq")
left_adc_data, right_adc_data = analyzer.run(left_dac_data, right_dac_data)

# Get a list of frequencies for the spectrum data. This is based on the params passed
# in analyzer.init() above
freq_array = analyzer.get_frequency_array();

# Get the spectrum amplitude arrays in dBV. The spectrum data bin count will 
# match the fft_size specified above
fft_array_left, fft_array_right = analyzer.get_amplitude_array("dbv")

# Compute the RMS energy from 900 to 1100 Hz
rms_left, rms_right = analyzer.compute_rms(900, 1100, "dbv")
print(f"rms left: {rms_left: 0.2f}dBV    rms right: {rms_right:.2f}dBV [900 to 1100 Hz]")

# Compute the RMS energy from 20 to 20 kHz
rms_left, rms_right = analyzer.compute_rms(20, 20000, "dbv")
print(f"rms left: {rms_left: 0.2f}dBV    rms right: {rms_right:.2f}dBV [20 to 20 kHz]")

# Plot the acquired ADC data
tsp = SeriesPlotter()
tsp.add_time_series(left_adc_data.get_main_buffer(), "Left", 512)
tsp.add_time_series(right_adc_data.get_buffer(), "Right", 512)
tsp.main_title = "ADC Data"
tsp.plot()

# Plot the acquired ADC spectrum
tsp2 = SeriesPlotter()
tsp2.add_freq_series(freq_array, fft_array_left, "Left")
tsp2.add_freq_series(freq_array, fft_array_right, "Right")
tsp2.main_title = "ADC Spectrum"
tsp2.plot()

# Release the interface
analyzer.cleanup()

```

With your QA40x connected, you can run the code above, and you should be greated with the following output:

![image](https://github.com/user-attachments/assets/38467217-cb9b-4db6-9888-f2e6577feded)


In the sample above, the left channel was connected in single-ended loopback, and the right channel had shorting blocks applied. In the code, you can see we created a waveform with two tones: a 0 dBV tone at 1 kHz, and a -50 dBV tone at 2 kHz. The amplitudes are correctly plotted. 

Next, an energy calc was done from 20 to 20 kHz and the returned number matched expectations. 

## What's Next

More functionality will be moved into the analyzer class, including the ability to calculate THD, THD+N, amplitude, frequency, etc. 

A plot class is included. The aim here is to get the basic plots down to a line of code. More work is needed there.

Chirp will be added to facilitate frequency response and phase plots. 

Stronger hinting and comments will hopefully enabled ChatGPT or other LLMs to become useful helpers for generating Jupyter notebooks with test. 

