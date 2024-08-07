import numpy as np
import usb1  # pip install libusb1
import struct
import atexit
import scipy.signal  # pip install scipy

from PyQa40x.registers import Registers
from PyQa40x.control import Control
from PyQa40x.stream import Stream
from PyQa40x.wave_sine import Wave
from PyQa40x.fft_processor import FFTProcessor
from PyQa40x.sig_proc import SigProc
from PyQa40x.helpers import *
from PyQa40x.analyzer_params import AnalyzerParams
from PyQa40x.bluetooth import BluetoothAudioDevice  

class Analyzer:
    def __init__(self):
        """
        Initializes the Analyzer class.
        """
        self.params: AnalyzerParams | None = None
        self.context: usb1.USBContext | None = None
        self.device: usb1.USBDeviceHandle | None = None
        self.registers: Registers | None = None
        self.control: Control | None = None
        self.stream: Stream | None = None
        self.cal_data: dict | None = None
        self.bt_device: BluetoothAudioDevice | None = None

    def init(self, sample_rate: int = 48000, max_input_level: int = 0, max_output_level: int = 18, 
             pre_buf: int = 2048, post_buf: int = 2048, fft_size: int = 16384, window_type: str = 'boxcar',
             amplitude_unit: str = "dbv", distortion_unit: str = "db", bt_device_name: str = '') -> AnalyzerParams:
        """
        Initializes the analyzer hardware with the specified parameters.

        Args:
            sample_rate (int): The sample rate for the device. Default is 48000. Valid values are 48000, 96000, 192000.
            max_input_level (int): Maximum input level in dBV. Valid values are 0, 6, 12, 18, 24, 30, 36, 42.
            max_output_level (int): Maximum output level in dBV. Valid values are 18, 8, -2, -12.
            pre_buf (int): Size of the pre-buffer.
            post_buf (int): Size of the post-buffer.
            fft_size (int): Size of the FFT.
            window_type (str): Type of window function to apply to the signal before FFT.
            amplitude_unit (str): Unit for amplitude measurements, default is "dbv".
            distortion_unit (str): Unit for distortion measurements, default is "db".
            bt_device_name (str): Name of the Bluetooth device to use. Default is empty.

        Returns:
            AnalyzerParams: A class instance containing the hardware and parameter settings.
        """
        self.params = AnalyzerParams(sample_rate, max_input_level, max_output_level, pre_buf, post_buf, fft_size, window_type)
        self.context = usb1.USBContext()

        # Attempt to open QA402 or QA403 device
        self.device = self.context.openByVendorIDAndProductID(0x16c0, 0x4e37)  # QA402
        if self.device is None:
            self.device = self.context.openByVendorIDAndProductID(0x16c0, 0x4e39)  # QA403
            if self.device is None:
                raise SystemExit("No QA402/QA403 analyzer found")
        self.device.resetDevice()
        self.device.claimInterface(0)

        self.registers = Registers(self.device)
        self.control = Control(self.registers)
        self.stream = Stream(self.context, self.device, self.registers)

        # Load calibration data
        self.cal_data = self.control.load_calibration()

        # Set input/output levels and sample rate
        self.control.set_input(max_input_level)
        self.control.set_output(max_output_level)
        self.control.set_samplerate(sample_rate)

        # Initialize Bluetooth device if a name is provided
        if bt_device_name:
            self.bt_device = BluetoothAudioDevice(bt_device_name, sample_rate)
            if self.bt_device.device_index is None:
                raise SystemExit("Failed to connect to the specified Bluetooth device")

        # Register cleanup function to be called on exit
        atexit.register(self.cleanup)

        return self.params

    def cleanup(self):
        """
        Releases the USB device interface and closes the context.
        """
        try:
            if self.device:
                self.device.releaseInterface(0)
            if self.context:
                self.context.close()
            if self.bt_device:
                self.bt_device.close()
        except Exception as e:
            print(f"An error occurred during cleanup: {e}")

    def send_receive(self, left_wave: Wave, right_wave: Wave) -> tuple[Wave, Wave]:
        """
        Sends DAC data to the device and receives ADC data.

        Args:
            left_wave (Wave): Wave instance with left channel DAC data.
            right_wave (Wave): Wave instance with right channel DAC data.

        Returns:
            tuple[Wave, Wave]: Tuple containing Wave instances with left and right channel ADC data.
        """
        left_dac_data = left_wave.get_buffer()
        right_dac_data = right_wave.get_buffer()

        # On QA402 and QA403, outputs need to be swapped
        left_dac_data, right_dac_data = right_dac_data, left_dac_data

        left_peak = np.max(left_dac_data)

        # Get calibration factors for outgoing data
        cal_dac_left, cal_dac_right = self.control.get_dac_cal(self.cal_data, self.params.max_output_level)

        # Convert to dBFS. Note the 3 dB adjustment--dBFS is peak, while dBV is RMS
        dac_dbfs_adjustment = 10 ** -((self.params.max_output_level + 3) / 20)

        # Apply calibration and dBFS scaling
        left_dac_data = left_dac_data * dac_dbfs_adjustment * cal_dac_left
        right_dac_data = right_dac_data * dac_dbfs_adjustment * cal_dac_right

        # Convert incoming doubles to float
        left_dac_data_float = left_dac_data.astype(np.float32)
        right_dac_data_float = right_dac_data.astype(np.float32)

        # Interleave the left and right channels
        interleaved_dac_data = np.empty((left_dac_data_float.size + right_dac_data_float.size,), dtype=np.float32)
        interleaved_dac_data[0::2] = left_dac_data_float
        interleaved_dac_data[1::2] = right_dac_data_float

        # Convert to bytes, multiplying by max int value
        max_int_value = 2 ** 31 - 1
        interleaved_dac_data = (interleaved_dac_data * max_int_value).astype(np.int32)

        # Pack the data into chunks of 16k bytes
        chunk_size = 16384  # 16k bytes
        num_ints_per_chunk = chunk_size // 4  # 32-bit ints, so 4 bytes per int
        total_chunks = len(interleaved_dac_data) // num_ints_per_chunk

        self.stream.start()

        # Play wave on Bluetooth if available
        bt_thread = None
        if self.bt_device:
            # Play both left and right waves on Bluetooth
            bt_thread = self.bt_device.play_wave_background(left_wave, right_wave)

        try:
            for i in range(total_chunks):
                chunk = interleaved_dac_data[i * num_ints_per_chunk:(i + 1) * num_ints_per_chunk]
                buffer = struct.pack('<%di' % len(chunk), *chunk)
                self.stream.write(buffer)
        finally:
            self.stream.stop()

        # Collect ADC data. This is bytes
        interleaved_adc_data = self.stream.collect_remaining_adc_data()
        
        # Wait for the Bluetooth thread to finish after processing
        if bt_thread:
            bt_thread.join()

        # Convert collected ADC data back to int
        interleaved_adc_data = np.frombuffer(interleaved_adc_data, dtype=np.int32)
        
        # Separate interleaved ADC data into left and right channels. This is int
        left_adc_data_int = interleaved_adc_data[0::2]
        right_adc_data_int = interleaved_adc_data[1::2]

        # Convert left and right channels back to double
        left_adc_data = left_adc_data_int.astype(np.float64) / max_int_value
        right_adc_data = right_adc_data_int.astype(np.float64) / max_int_value

        # Get calibration factors for incoming data
        cal_adc_left, cal_adc_right = self.control.get_adc_cal(self.cal_data, self.params.max_input_level)

        # Convert from dBFS to dBV. Note the 6 dB factor--the ADC is differential
        adc_dbfs_correction = 10 ** ((self.params.max_input_level - 6) / 20)

        # Apply calibration and dBFS scaling
        left_adc_data = left_adc_data * cal_adc_left * adc_dbfs_correction
        right_adc_data = right_adc_data * cal_adc_right * adc_dbfs_correction

        # Ensure the buffer matches the expected shape. Is this needed?
        expected_shape = (self.params.pre_buf + self.params.fft_size + self.params.post_buf,)
        if left_adc_data.shape[0] != expected_shape[0]:
            left_adc_data = np.resize(left_adc_data, expected_shape)
        if right_adc_data.shape[0] != expected_shape[0]:
            right_adc_data = np.resize(right_adc_data, expected_shape)

        # Create instances of Wave with the full buffers
        left_wave = Wave(self.params)
        right_wave = Wave(self.params)
        left_wave.set_buffer(left_adc_data)
        right_wave.set_buffer(right_adc_data)

        return left_wave, right_wave
    

    @staticmethod
    def list_audio_devices_by_sample_rate(target_sample_rate: int):
        """List all available audio devices that support a specific sample rate by calling the method from BluetoothAudioDevice."""
        BluetoothAudioDevice.list_audio_devices_by_sample_rate(target_sample_rate)


