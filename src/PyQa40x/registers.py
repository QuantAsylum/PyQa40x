import struct
import usb1

class Registers:
    """
    Class to handle low-level register read/write operations with the USB device.
    """
    def __init__(self, device):
        """
        Initializes the Registers class with a USB device.

        Args:
            device (usb1.USBDeviceHandle): The USB device handle.
        """
        self.endpoint_read = usb1.ENDPOINT_IN | 0x01  # EP1 in
        self.endpoint_write = usb1.ENDPOINT_OUT | 0x01  # EP1 out
        self.device = device

    def read(self, reg):
        """
        Reads a 32-bit value from a specified register.

        Args:
            reg (int): The register address to read from.

        Returns:
            int: The 32-bit value read from the register.
        """
        self.write(0x80 | reg, 0)  # Write the address, with MSB set
        data = self.device.bulkRead(self.endpoint_read, 4, 1000)  # Read result
        (val,) = struct.unpack('>I', data)  # 32-bit big endian
        return val

    def write(self, reg, val):
        """
        Writes a 32-bit value to a specified register.

        Args:
            reg (int): The register address to write to.
            val (int): The 32-bit value to write to the register.
        """
        buf = struct.pack('>BI', reg, val)  # 8-bit address and 32-bit big endian value
        self.device.bulkWrite(self.endpoint_write, buf, 1000)
