import time
import struct

class Control:
    """
    Class to manage device settings, including output and input gain, sample rate, and calibration data.
    """
    def __init__(self, registers):
        """
        Initializes the Control class with a Registers instance.

        Args:
            registers (Registers): An instance of the Registers class for low-level register access.
        """
        self.registers = registers
        self.output2reg = {18: 3, 8: 2, -2: 1, -12: 0}
        self.input2reg = {0: 0, 6: 1, 12: 2, 18: 3, 24: 4, 30: 5, 36: 6, 42: 7}
        self.samplerate2reg = {48000: 0, 96000: 1, 192000: 2}

    def set_output(self, gain):
        """
        Sets the output gain of the device.

        Args:
            gain (int): The output gain in dBV. Valid values are 18, 8, -2, -12.
        """
        val = self.output2reg[gain]
        self.registers.write(6, val)

    def set_input(self, gain):
        """
        Sets the input gain of the device.

        Args:
            gain (int): The input gain in dBV. Valid values are 0, 6, 12, 18, 24, 30, 36, 42.
        """
        val = self.input2reg[gain]
        self.registers.write(5, val)

    def set_samplerate(self, rate):
        """
        Sets the sample rate of the device.

        Args:
            rate (int): The sample rate in Hz. Valid values are 48000, 96000, 192000.
        """
        val = self.samplerate2reg[rate]
        self.registers.write(9, val)
        time.sleep(0.1)  # A small delay to ensure the sample rate is set

    def load_calibration(self):
        """
        Loads the calibration data from the device.

        Returns:
            bytearray: The calibration data.
        """
        self.registers.write(0xd, 0x10)
        page_size = 512
        cal_data = bytearray(page_size)

        for i in range(page_size // 4):
            d = self.registers.read(0x19)
            array = struct.pack('<I', d)
            cal_data[i * 4:(i + 1) * 4] = array

        return cal_data

    def get_adc_cal(self, cal_data, full_scale_input_level):
        """
        Gets the ADC calibration factors for a specified input level.

        Args:
            cal_data (bytearray): The calibration data.
            full_scale_input_level (int): The full-scale input level in dBV.

        Returns:
            tuple: Calibration factors for left and right ADC channels.
        """
        offsets = {0: 24, 6: 36, 12: 48, 18: 60, 24: 72, 30: 84, 36: 96, 42: 108}

        if full_scale_input_level not in offsets:
            raise ValueError("Invalid input level. Must be one of 0, 6, 12, 18, 24, 30, 36, 42.")

        left_offset = offsets[full_scale_input_level]
        right_offset = left_offset + 6  # Right level is 6 bytes after left level

        left_level, left_value = struct.unpack_from('<hf', cal_data, left_offset)
        right_level, right_value = struct.unpack_from('<hf', cal_data, right_offset)

        left_value = 10 ** (left_value / 20)
        right_value = 10 ** (right_value / 20)

        return left_value, right_value

    def get_dac_cal(self, cal_data, full_scale_output_level):
        """
        Gets the DAC calibration factors for a specified output level.

        Args:
            cal_data (bytearray): The calibration data.
            full_scale_output_level (int): The full-scale output level in dBV.

        Returns:
            tuple: Calibration factors for left and right DAC channels.
        """
        offsets = {18: 156, 8: 144, -2: 132, -12: 120}

        if full_scale_output_level not in offsets:
            raise ValueError("Invalid output level. Must be one of 18, 8, -2, -12.")

        left_offset = offsets[full_scale_output_level]
        right_offset = left_offset + 6  # Right level is 6 bytes after left level

        left_level, left_value = struct.unpack_from('<hf', cal_data, left_offset)
        right_level, right_value = struct.unpack_from('<hf', cal_data, right_offset)

        left_value = 10 ** (left_value / 20)
        right_value = 10 ** (right_value / 20)

        return left_value, right_value

    def dump_calibration_data(self, cal_data):
        """
        Dumps the calibration data for debugging purposes.

        Args:
            cal_data (bytearray): The calibration data.
        """
        hex_data = ' '.join(f'{byte:02X}' for byte in cal_data)
        print(hex_data)

        left_value, right_value = self.get_adc_cal(cal_data, 42)
        print(f"ADC Left level: {left_value}, Right level: {right_value}")

        left_value, right_value = self.get_dac_cal(cal_data, -2)
        print(f"DAC Left level: {left_value}, Right level: {right_value}")
