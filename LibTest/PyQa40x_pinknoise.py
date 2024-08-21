import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PyQa40x.analyzer import Analyzer
from PyQa40x.wave import Wave

# Generate a periodic arbitrary spectrum, capture it and performs smoothing. What is
# notable here is we can generate any type of noise we want (the example is showing
# 5 dB/decade rolloff--pink noise is 10 dB/decade). And then, the noise is 
# then played out of the DAC and captured by the ADC. A single acquisition with 
# a bit of smoothing can deliver spectrum response curves very quickly. If 
# not using the QA403 (use_qa40x=False), you can experiment with different step sizes (overlap) 
# to help with noise. However, as you'll see, overlap doesn't play much of a role 
# in reducing noise. The big win comes from smoothing.

# Parameters
fft_size = 32768
sample_rate = 48000
octave_fraction = 6  # Specify the octave fraction for smoothing (e.g., 1/3 octave). 0 = no smoothing
overlap = 0.95  # Set overlap to 95%
step_size = int(fft_size * (1 - overlap))
use_qa40x = True  # Set to True to use channel noise, False to use np.random.normal

def smooth_spectrum(frequencies, spectrum, octave_fraction=3):
    """
    Smooth the frequency spectrum using octave band smoothing.
    
    :param frequencies: Array of frequency bins.
    :param spectrum: Array of spectrum magnitudes (linear scale).
    :param octave_fraction: Fraction of octave for smoothing (e.g., 1/3 octave).
    :return: Smoothed spectrum.
    """
    if octave_fraction == 0:
        return spectrum.copy()

    smoothed_spectrum = np.zeros_like(spectrum)
    num_points = len(frequencies)
    
    for i in range(num_points):
        # Calculate the bounds of the current band
        f_center = frequencies[i]
        f_lower = f_center / (2**(1/(2*octave_fraction)))
        f_upper = f_center * (2**(1/(2*octave_fraction)))
        
        # Find indices corresponding to the current band
        band_indices = np.where((frequencies >= f_lower) & (frequencies <= f_upper))[0]
        
        # Average the spectrum values within the current band
        smoothed_spectrum[i] = np.mean(spectrum[band_indices])
    
    return smoothed_spectrum

frequencies = np.fft.fftfreq(fft_size, d=1/sample_rate)

# Generate frequency domain signal. 
mag = np.zeros(fft_size // 2 + 1)

# We want each frequency to have a random phase, otherwise
# there's lots of cancelation occuring.
phase = np.random.uniform(0, 2 * np.pi, fft_size // 2 + 1)


# Generate a spectrum that has a level of -50 dB outside
# the range of 10 to 10 kHz. And from 10 Hz up to 10, 
# start at 20 dB at 10 Hz, dropping 5 dB/decade up to 10k
for i, freq in enumerate(frequencies[:fft_size // 2 + 1]):
    if freq < 10 or freq > 10000:
        mag[i] = 10**(-50/20)  # -50 dB in linear scale
    else:
        mag[i] = 10**((20 - 5 * np.log10(freq/10))/20)
        

# Construct the complex signal in the frequency domain (ensure symmetry)
freq_domain_signal = mag * np.exp(1j * phase)
freq_domain_signal = np.concatenate((freq_domain_signal, np.conj(freq_domain_signal[-2:0:-1])))

# Plot the original frequency response
plt.figure()
plt.plot(frequencies[:fft_size//2], 20 * np.log10(mag[:fft_size//2]))
plt.xscale('log')
plt.ylim(-60, 20)  # Fix y-axis range from -60 dB to 20 dB
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Original Frequency Domain Signal')
plt.grid(True)
plt.show()

# Perform IFFT to get the real time domain signal
time_domain_signal = np.fft.ifft(freq_domain_signal).real

# Concatenate the original signal with itself
DAC = np.concatenate((time_domain_signal, time_domain_signal))

# Simulate sending DAC over a wire and capturing it as ADC
ADC = DAC.copy()  # Replace this with actual ADC acquisition code

# Add noise to the captured data based on the use_qa40x flag
if use_qa40x:
    # Create an Analyzer instance
    analyzer = Analyzer()
    # Initialize the analyzer with desired parameters
    params = analyzer.init(sample_rate=sample_rate, max_input_level=0, max_output_level=8, fft_size=fft_size, window_type="flattop")
    dac = Wave(params)
    dac.set_buffer(np.concatenate( (np.zeros(params.pre_buf), time_domain_signal, np.zeros(params.post_buf))))
    adc, _ = analyzer.send_receive(dac, dac)
    DAC = time_domain_signal
    ADC = adc.get_main_buffer()
    analyzer.cleanup()
else:
    # Add Gaussian noise via np.random.normal
    noise = np.random.normal(0, 0.0001, len(ADC))  # Adjusted standard deviation for noise level
    ADC += noise

# Plot the DAC and noisy ADC signals
plt.figure()
plt.plot(DAC)
plt.title('DAC Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(ADC)
plt.title('Noisy ADC Signal' if not use_qa40x else 'ADC Signal with Channel Noise')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Calculate the number of FFTs possible.  Note that if using the QA40x
# hardware, it will always be a single FFT
num_ffts = (len(ADC) - fft_size) // step_size + 1

# Perform FFTs with the specified overlap.
averaged_spectrum = np.zeros(fft_size // 2 + 1)

for i in range(num_ffts):
    start_index = i * step_size
    end_index = start_index + fft_size
    if end_index > len(ADC):
        break
    segment = ADC[start_index:end_index]
    spectrum = np.abs(np.fft.fft(segment)[:fft_size // 2 + 1])
    averaged_spectrum += spectrum

averaged_spectrum /= num_ffts

# Smooth the spectrum using the specified octave fraction
smoothed_spectrum = smooth_spectrum(frequencies[:fft_size//2], averaged_spectrum[:fft_size//2], octave_fraction=octave_fraction)

# Plot the averaged and smoothed frequency response in dB
plt.figure()
plt.plot(frequencies[:fft_size//2], 20 * np.log10(averaged_spectrum[:fft_size//2]), label='Averaged')
plt.plot(frequencies[:fft_size//2], 20 * np.log10(smoothed_spectrum), label=f'Smoothed (1/{octave_fraction} Octave)', linestyle='--')
plt.xscale('log')
plt.ylim(-60, 20)  # Fix y-axis range from -60 dB to 20 dB
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title(f'Averaged and Smoothed Frequency Domain Signal ({num_ffts} FFTs, {int(overlap * 100)}% Overlap)')
plt.legend()
plt.grid(True)
plt.show()

