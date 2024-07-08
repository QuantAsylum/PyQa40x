import threading
import queue
import usb1
import numpy as np

class Stream:
    """
    Class to manage buffer streaming, with event worker thread.
    """
    def __init__(self, context, device, registers):
        """
        Initializes the Stream class.

        Args:
            context (usb1.USBContext): The USB context.
            device (usb1.USBDeviceHandle): The USB device handle.
            registers (Registers): An instance of the Registers class.
        """
        self.context = context
        self.device = device
        self.registers = registers
        self.endpoint_read = usb1.ENDPOINT_IN | 0x02  # EP2 in
        self.endpoint_write = usb1.ENDPOINT_OUT | 0x02  # EP2 out
        self.dacQueue = queue.Queue(maxsize=5)  # Max. 5 overlapping buffers in flight, block on more
        self.adcQueue = queue.Queue()  # Unlimited queue for received data buffers
        self.transfer_helper = usb1.USBTransferHelper()  # Use the callback dispatcher
        self.transfer_helper.setEventCallback(usb1.TRANSFER_COMPLETED, self.callback)  # Set ours
        self.received_data = bytearray()  # Collection of received data bytes
        self.thread = None
        self.running = False

    def start(self):
        """
        Starts the streaming and spawns the worker thread.
        """
        self.thread = threading.Thread(target=self.worker)
        self.running = True
        self.thread.start()
        self.registers.write(8, 0x05)  # Start streaming

    def stop(self):
        """
        Stops the streaming and ends the worker thread.
        """
        self.running = False
        self.thread.join()
        self.registers.write(8, 0x00)  # Stop streaming

    def write(self, buffer):
        """
        Adds a buffer to the playback queue.

        Args:
            buffer (bytes): The data buffer to be written.
        """
        transfer = self.device.getTransfer()
        transfer.setBulk(self.endpoint_write, buffer, self.transfer_helper, None, 1000)
        transfer.submit()  # Asynchronous transfer
        self.dacQueue.put(transfer)  # It doesn't matter what we put in here

        # Submit a USB bulk transfer to read
        read_transfer = self.device.getTransfer()
        read_transfer.setBulk(self.endpoint_read, 16384, self.transfer_helper, None, 1000)
        read_transfer.submit()  # Asynchronous transfer
        self.adcQueue.put(read_transfer)  # It doesn't matter what we put in here

    def worker(self):
        """
        Event loop for the asynchronous transfers.
        """
        while self.running or not (self.dacQueue.empty() and self.adcQueue.empty()):  # Play until the last
            self.context.handleEvents()

    def callback(self, transfer):
        """
        Callback of the worker thread to handle completed transfers.

        Args:
            transfer (usb1.USBTransfer): The USB transfer that has completed.
        """
        if transfer.getEndpoint() == self.endpoint_read:
            self.received_data.extend(transfer.getBuffer())  # Collect received data bytes
            self.adcQueue.get()  # Unblock the producer (should pop same transfer)
        else:
            self.dacQueue.get()  # Unblock the producer (should pop same transfer)

    def collect_remaining_adc_data(self):
        """
        Waits for all remaining ADC transfers to complete and returns the collected data.

        Returns:
            bytearray: The collected ADC data.
        """
        # Wait for all remaining ADC transfers to complete
        while not self.adcQueue.empty():
            self.context.handleEvents()
        return self.received_data
