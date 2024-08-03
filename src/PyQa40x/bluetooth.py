import pyaudio
import threading
import numpy as np
from PyQa40x import Wave

class BluetoothAudioDevice:
    def __init__(self, device_name: str, sample_rate: int):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.device_index = None
        self.p = pyaudio.PyAudio()

        # Attempt to find and open the device
        self.device_index = self._find_device()
        if self.device_index is None:
            print("Failed to find the audio device with the required sample rate.")
    
    def _find_device(self):
        """Find and return the index of the target audio device if it supports the required sample rate."""
        target_device_index = None
        print("Enumerating audio devices...\n")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            
            if self.device_name in info['name']:
                print(f"  Checking device '{info['name']}' for sample rate {self.sample_rate} Hz...")
                try:
                    # Attempt to open a stream to check if the sample rate is supported
                    stream = self.p.open(format=pyaudio.paFloat32,
                                         channels=1,
                                         rate=self.sample_rate,
                                         output=True,
                                         output_device_index=i)
                    stream.close()
                    print(f"  -> Device {i} supports the required sample rate.")
                    if target_device_index is None:
                        target_device_index = i
                except Exception as e:
                    print(f"  -> Device {i} does not support the required sample rate. ({str(e)})")
        return target_device_index

    def play_wave(self, wave: Wave):
        """Play a Wave object in the background."""
        if self.device_index is None:
            print("No valid audio device found. Cannot play wave.")
            return
        
        # Start playing the wave in a separate thread
        thread = threading.Thread(target=self._play_wave_background, args=(wave, self.sample_rate, self.device_index))
        thread.start()
        return thread

    def _play_wave_background(self, wave: Wave, sample_rate: int, device_index: int):
        """Play a Wave object."""
        stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=sample_rate,
                             output=True,
                             output_device_index=device_index)

        actual_sample_rate = stream._rate
        print(f"Requested Sample Rate: {sample_rate}")
        print(f"Actual Sample Rate: {actual_sample_rate}")

        # Convert wave data to float32 format and play
        stream.write(wave.astype(np.float32).tobytes())
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
                                channels=1,
                                rate=target_sample_rate,
                                output=True,
                                output_device_index=i)
                stream.close()
                print(f"Device {i}: {info['name']} supports {target_sample_rate} Hz")
            except Exception:
                # Device does not support the specified sample rate
                continue
        p.terminate()


