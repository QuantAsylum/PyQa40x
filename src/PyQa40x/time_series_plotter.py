import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TimeSeriesPlotter:
    def __init__(self, num_columns=2, main_title="", main_title_fontsize=16):
        self.num_columns = num_columns
        self.time_series_traces = []
        self.main_title = main_title
        self.main_title_fontsize = main_title_fontsize
    
    def add_time_series(self, signal, label, num_samples=0):
        self.time_series_traces.append({
            'signal': signal,
            'label': label,
            'num_samples': num_samples
        })
    
    def plot(self):
        num_traces = len(self.time_series_traces)
        num_rows = (num_traces + self.num_columns - 1) // self.num_columns
        fig, axs = plt.subplots(num_rows, self.num_columns, figsize=(14, 6 * num_rows))

        for i, trace in enumerate(self.time_series_traces):
            ax = axs.flat[i] if num_rows > 1 else axs[i]
            signal = trace['signal']
            label = trace['label']
            num_samples = trace['num_samples']
            if num_samples > 0:
                signal = signal[:num_samples]

            ax.plot(signal)
            ax.set_title(label)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude (Volts)')

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, linewidth=1, edgecolor='black', facecolor='none')
        fig.patches.append(rect)
        
        plt.show()
