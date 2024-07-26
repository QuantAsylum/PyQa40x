import numpy as np
import matplotlib.pyplot as plt

class SeriesPlotter:
    def __init__(self, num_columns=2, main_title="", main_title_fontsize=16):
        self.num_columns = num_columns
        self.rows = [[]]
        self.main_title = main_title
        self.main_title_fontsize = main_title_fontsize

    def add_time_series(self, signal, label, signal_right=None, num_samples=0, units="Volts", units_right="Volts", ymin=None, ymax=None, ymin_right=None, ymax_right=None, xmin=None, xmax=None, logx=False):
        self.rows[-1].append({
            'type': 'time',
            'signal': signal,
            'signal_right': signal_right,
            'label': label,
            'num_samples': num_samples,
            'units': units,
            'units_right': units_right,
            'ymin': ymin,
            'ymax': ymax,
            'ymin_right': ymin_right,
            'ymax_right': ymax_right,
            'xmin': xmin,
            'xmax': xmax,
            'logx': logx
        })

    def add_freq_series(self, freqs, magnitudes, label, magnitudes_right=None, num_samples=0, units="dBV", units_right="dBV", ymin=None, ymax=None, ymin_right=None, ymax_right=None, xmin=None, xmax=None, logx=False):
        if logx:
            xmin = xmin if xmin is not None else 20
            xmax = xmax if xmax is not None else 20000

        self.rows[-1].append({
            'type': 'freq',
            'freqs': freqs,
            'magnitudes': magnitudes,
            'magnitudes_right': magnitudes_right,
            'label': label,
            'num_samples': num_samples,
            'units': units,
            'units_right': units_right,
            'ymin': ymin,
            'ymax': ymax,
            'ymin_right': ymin_right,
            'ymax_right': ymax_right,
            'xmin': xmin,
            'xmax': xmax,
            'logx': logx
        })

    def newrow(self):
        if self.rows[-1]:
            self.rows.append([])

    def plot(self, block=True):
        mosaic_layout = []
        for row in self.rows:
            if not row:  # Skip empty rows
                continue
            # Handle rows with more elements than num_columns
            while len(row) > self.num_columns:
                mosaic_layout.append([trace['label'] for trace in row[:self.num_columns]])
                row = row[self.num_columns:]
            # Evenly distribute the elements in the row
            num_elements = len(row)
            if num_elements < self.num_columns:
                span_each = self.num_columns // num_elements
                remainder = self.num_columns % num_elements
                new_row = []
                for i in range(num_elements):
                    span = span_each + (1 if i < remainder else 0)
                    new_row.extend([row[i]['label']] * span)
                mosaic_layout.append(new_row)
            else:
                mosaic_layout.append([trace['label'] for trace in row])

        label_to_trace = {trace['label']: trace for row in self.rows for trace in row}

        fig, axd = plt.subplot_mosaic(mosaic_layout, figsize=(5 * self.num_columns, 4 * len(mosaic_layout)))

        for label, ax in axd.items():
            if label is not None:
                trace = label_to_trace[label]
                if trace['type'] == 'time':
                    signal = trace['signal']
                    signal_right = trace['signal_right']
                    num_samples = trace['num_samples']
                    units = trace['units']
                    units_right = trace['units_right']
                    ymin = trace['ymin']
                    ymax = trace['ymax']
                    ymin_right = trace['ymin_right']
                    ymax_right = trace['ymax_right']
                    xmin = trace['xmin']
                    xmax = trace['xmax']
                    logx = trace['logx']
                    if num_samples > 0:
                        signal = signal[:num_samples]
                        if signal_right is not None:
                            signal_right = signal_right[:num_samples]

                    ax.plot(signal, label=label)
                    ax.set_title(label)
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel(f'Amplitude ({units})')
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    if xmin is not None and xmax is not None:
                        ax.set_xlim(xmin, xmax)
                    if logx:
                        ax.set_xscale('log')

                    if signal_right is not None:
                        ax_right = ax.twinx()
                        ax_right.plot(signal_right, 'r', label=f'{label} Right')
                        ax_right.set_ylabel(f'Amplitude ({units_right})', color='r')
                        if ymin_right is not None and ymax_right is not None:
                            ax_right.set_ylim(ymin_right, ymax_right)

                elif trace['type'] == 'freq':
                    freqs = trace['freqs']
                    magnitudes = trace['magnitudes']
                    magnitudes_right = trace['magnitudes_right']
                    units = trace['units']
                    units_right = trace['units_right']
                    ymin = trace['ymin']
                    ymax = trace['ymax']
                    ymin_right = trace['ymin_right']
                    ymax_right = trace['ymax_right']
                    xmin = trace['xmin']
                    xmax = trace['xmax']
                    logx = trace['logx']
                    if num_samples > 0:
                        freqs = freqs[:num_samples]
                        magnitudes = magnitudes[:num_samples]
                        if magnitudes_right is not None:
                            magnitudes_right = magnitudes_right[:num_samples]

                    ax.plot(freqs, magnitudes, label=label)
                    ax.set_title(label)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel(f'Magnitude ({units})')
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    if xmin is not None and xmax is not None:
                        ax.set_xlim(xmin, xmax)
                    if logx:
                        ax.set_xscale('log')

                    if magnitudes_right is not None:
                        ax_right = ax.twinx()
                        ax_right.plot(freqs, magnitudes_right, 'r', label=f'{label} Right')
                        ax_right.set_ylabel(f'Magnitude ({units_right})', color='r')
                        if ymin_right is not None and ymax_right is not None:
                            ax_right.set_ylim(ymin_right, ymax_right)

        if self.main_title:
            fig.suptitle(self.main_title, fontsize=self.main_title_fontsize)
            fig.subplots_adjust(top=0.95)  # Adjust the top spacing to reduce whitespace

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to reduce whitespace
        plt.show(block=block)


