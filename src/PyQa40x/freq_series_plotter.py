import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class FreqSeriesPlotter:
    def __init__(self, params, num_columns=2, main_title="", main_title_fontsize=16, log_x_axis=True, units='dBV'):
        self.params = params
        self.num_columns = num_columns
        self.main_title = main_title
        self.main_title_fontsize = main_title_fontsize
        self.log_x_axis = log_x_axis
        self.units = units
        self.freq_series_traces = []

    def add_freq_series(self, signal: np.ndarray, label: str):
        self.freq_series_traces.append({'signal': signal, 'label': label})

    def plot(self):
        num_rows = (len(self.freq_series_traces) + self.num_columns - 1) // self.num_columns
        fig, axs = plt.subplots(num_rows, self.num_columns, figsize=(14, num_rows * 3.5))

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)

        for i, trace in enumerate(self.freq_series_traces):
            ax = axs.flat[i] if num_rows > 1 else axs[i]
            signal = trace['signal']
            label = trace['label']
            fft_freqs = np.fft.rfftfreq(len(signal) * 2 - 1, 1 / self.params.sample_rate)

            if self.units == 'dBV':
                log_magnitude = 20 * np.log10(signal)
            elif self.units == 'dBu':
                log_magnitude = 20 * np.log10(signal) + 2.2
            else:
                raise ValueError("Unsupported units. Use 'dBV' or 'dBu'.")

            ax.plot(fft_freqs, log_magnitude)
            ax.set_title(label)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(f"Amplitude ({self.units})")
            ax.grid(True)
            ax.set_ylim(-150, 10)
            if self.log_x_axis:
                ax.set_xscale('log')
                ax.set_xlim(20, 20000)
                ax.set_xticks([20, 100, 1000, 10000])
                ax.set_xticklabels(['20', '100', '1k', '10k'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, linewidth=1, edgecolor='black', facecolor='none')
        fig.patches.append(rect)

        plt.show()
