import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SeriesPlotter:
    def __init__(self, num_columns=2, main_title="", main_title_fontsize=16):
        self.num_columns = num_columns
        self.time_series_traces = []
        self.freq_series_traces = []
        self.main_title = main_title
        self.main_title_fontsize = main_title_fontsize
    
    def add_time_series(self, signal, label, num_samples=0, units="Volts"):
        self.time_series_traces.append({
            'signal': signal,
            'label': label,
            'num_samples': num_samples,
            'units': units
        })
    
    def add_freq_series(self, freqs, magnitudes, label, num_samples=0, units="dBV"):
        self.freq_series_traces.append({
            'freqs': freqs,
            'magnitudes': magnitudes,
            'label': label,
            'num_samples': num_samples,
            'units': units
        })
    
    def plot(self, block=True):
        num_time_traces = len(self.time_series_traces)
        num_freq_traces = len(self.freq_series_traces)
        num_traces = num_time_traces + num_freq_traces
        num_rows = (num_traces + self.num_columns - 1) // self.num_columns

        # Adjust the number of columns to 1 if there's only one trace
        num_columns = self.num_columns if num_traces > 1 else 1
        
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(14, 6 * num_rows))

        # Flatten axs if it's not already flat (for single column/row cases)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        index = 0
        for trace in self.time_series_traces:
            ax = axs.flat[index] if num_rows > 1 or num_columns > 1 else axs[0]
            signal = trace['signal']
            label = trace['label']
            num_samples = trace['num_samples']
            units = trace['units']
            if num_samples > 0:
                signal = signal[:num_samples]

            ax.plot(signal)
            ax.set_title(label)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'Amplitude ({units})')
            index += 1

        for trace in self.freq_series_traces:
            ax = axs.flat[index] if num_rows > 1 or num_columns > 1 else axs[0]
            freqs = trace['freqs']
            magnitudes = trace['magnitudes']
            label = trace['label']
            num_samples = trace['num_samples']
            units = trace['units']
            if num_samples > 0:
                freqs = freqs[:num_samples]
                magnitudes = magnitudes[:num_samples]

            ax.plot(freqs, magnitudes)
            ax.set_title(label)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'Magnitude ({units})')
            index += 1

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, linewidth=1, edgecolor='black', facecolor='none')
        fig.patches.append(rect)
        
        plt.show(block=block)