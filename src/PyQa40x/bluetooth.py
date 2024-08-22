import pyaudio # pip install pyaudio
import threading
import numpy as np
from PyQa40x.wave import Wave

class BluetoothAudioDevice:
    def __init__(self, device_name: str, sample_rate: int, debug: bool = False):
        """
        Initializes the BluetoothAudioDevice.

        Args:
            device_name (str): Name of the target Bluetooth audio device.
            sample_rate (int): Sample rate for playback.
            debug (bool): If True, print detailed debug information. Default is False.
        """
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.debug = debug
        self.device_index = None
        self.p = pyaudio.PyAudio()

        # Attempt to find and open the device
        self.device_index = self._find_device()
        if self.device_index is None:
            print(f"Failed to open audio device '{self.device_name}' with the required sample rate {self.sample_rate} Hz.")

    def _find_device(self):
        """Find and return the index of the target audio device if it supports the required sample rate."""
        target_device_index = None
        if self.debug:
            print("Enumerating audio devices...\n")

        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if self.debug:
                print(f"Device {i}: {info['name']}")
            
            if self.device_name in info['name']:
                if self.debug:
                    print(f"  Checking device '{info['name']}' for sample rate {self.sample_rate} Hz...")
                try:
                    # Attempt to open a stream to check if the sample rate is supported
                    stream = self.p.open(format=pyaudio.paFloat32,
                                         channels=2,
                                         rate=self.sample_rate,
                                         output=True,
                                         output_device_index=i)
                    stream.close()
                    if self.debug:
                        print(f"  -> Device {i} supports the required sample rate.")
                    if target_device_index is None:
                        target_device_index = i
                except Exception as e:
                    if self.debug:
                        print(f"  -> Device {i} does not support the required sample rate. ({str(e)})")
        return target_device_index

    def play_wave_background(self, left_wave: Wave, right_wave: Wave):
        """Play left and right Wave objects in the background."""
        if self.device_index is None:
            print("No valid audio device found. Cannot play wave.")
            return None
        
        # Start playing the wave in a separate thread
        thread = threading.Thread(target=self._play_wave, args=(left_wave, right_wave, self.sample_rate, self.device_index))
        thread.start()
        return thread

    def _play_wave(self, left_wave: Wave, right_wave: Wave, sample_rate: int, device_index: int):
        """Play a stereo Wave object."""
        stream = self.p.open(format=pyaudio.paFloat32,
                             channels=2,  # Stereo playback
                             rate=sample_rate,
                             output=True,
                             output_device_index=device_index)

        actual_sample_rate = stream._rate
        if self.debug:
            print(f"Requested Sample Rate: {sample_rate}")
            print(f"Actual Sample Rate: {actual_sample_rate}")

        # Get buffers for left and right waves
        left_buffer = left_wave.get_buffer().astype(np.float32)
        right_buffer = right_wave.get_buffer().astype(np.float32)

        # Ensure both buffers are the same length
        min_length = min(len(left_buffer), len(right_buffer))
        left_buffer = left_buffer[:min_length]
        right_buffer = right_buffer[:min_length]

        # Interleave left and right buffers for stereo output
        interleaved = np.empty((min_length * 2,), dtype=np.float32)
        interleaved[0::2] = left_buffer
        interleaved[1::2] = right_buffer

        # Convert wave data to float32 format and play
        stream.write(interleaved.tobytes())
        stream.stop_stream()
        stream.close()

    def close(self):
        """Terminate the PyAudio instance."""
        self.p.terminate()

    @staticmethod
    def list_audio_devices():
        """List all available audio devices by name."""
        p = pyaudio.PyAudio()
        print("Available audio devices:\n")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
        p.terminate()

    @staticmethod
    def list_audio_devices_by_sample_rate(target_sample_rate: int):
        """List all available audio devices that support a specific sample rate."""
        p = pyaudio.PyAudio()
        print(f"Audio devices supporting {target_sample_rate} Hz sample rate:\n")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            try:
                # Check if the device supports the specified sample rate
                stream = p.open(format=pyaudio.paFloat32,
                                channels=2,  # Check for stereo support
                                rate=target_sample_rate,
                                output=True,
                                output_device_index=i)
                stream.close()
                print(f"Device {i}: {info['name']} supports {target_sample_rate} Hz")
            except Exception:
                # Device does not support the specified sample rate
                continue
        p.terminate()
