PyQa40x allows you to run Python code to communicate with the QA40x hardware. No intermediate program is needed, as the PyQa40x lib understands how to use the calibration data stored inside the QA40x hardware.

The library is very much in flux, and support might be sporadic. The PyQa40x lib is one of several ways to [use the QA40x hardware](https://github.com/QuantAsylum/QA40x/wiki/QA40x-API). There are now 5 ways to interact with your QA40x hardware:

1) Using the QA40x application. This is the most-supported path. An addition to making a host of basic audio measurements, the QA40x app can also do swept plots, which is useful for plotting thinks like THD versus Level, for example. The application is located [here](https://github.com/QuantAsylum/QA40x/releases)
2) Tractor. Tractor allows you create scripts to drive automated testing. You can establish pass/fail limits, allowing an operating with limited skills to test products in a production environment. Tractor works in conjuction with the QA40x application to "drive" your tests. The application is located [here](https://github.com/QuantAsylum/Tractor/releases/tag/v1.101)
3) REST. REST is a common way of interacting with other entities on the internet, and every modern language provides support for implementing REST. The QA40x application handles the REST communication, and allows you to use GET/PUT/POST to control the measurements. A test program in C# is located [here](https://github.com/QuantAsylum/QA402_REST_TEST). More info on the Wiki is located [here](https://github.com/QuantAsylum/QA40x/wiki/QA40x-API)
4) Bare Metal. If you'd like to see how to communicate with the QA40x hardware directly with c#, you could look at the repo located [here](https://github.com/QuantAsylum/QA40x_BareMetal)
5) ASIO. An ASIO driver for the QA403/QA402/QA401 is available [here](https://github.com/dechamps/ASIO401), allowing you to use the QA40x hardware with common audio-processing applications.
6) PyQa40x. This library is an attempt to streamline a lot of common processing operations, while handling some of the more difficult topics such as normalization and windowing. The aim here is to facilitate an environment to digital signal processing development using absolute values instead of relative values. What this means is if you specify a 1Vrms sine way, the output of of the QA40x hardware will be 1Vrms.

The rest of this document focuses on running the PyQa40x examples under Jupyterlab Desktop.

## Getting up and running

If you are looking for a simple starting point for running Python, Jupyterlab Desktop is probably it. It delivers a nice editing environment, includes its own Python installer, and delivers beautiful plots. 

### Install JupyterLab Desktop
There are a lot of flavors of Jupyter. JupyterLab Desktop is the cross-platform desktop application for JupyterLab, and it is probably the quickest and easiest way to get started with Jupyter notebooks on your local machine.

Go to the JupyterLab Desktop page on Github located [here](https://github.com/jupyterlab/jupyterlab-desktop)

Scroll down to the ```Installation``` section and select the installer for your particular operating system.

Once the application has installed, upon first run, you'll see an option to install Python at the base of the Jupyter home screen.

Once Python has installed, go to the Github page for the [PyQa40x lib](https://github.com/QuantAsylum/PyQa40x) and locate the green ```Code``` button. Click that and download a zip of the library. Unzip that into the directory of your choice.

Re-launch the JupyterLab Desktop app, and from the main screen, select "Open Folder." Create a new folder on your drive off of MyDocs, and name it ```JupyterLabAudioTest``` or similar. 

![image](https://github.com/user-attachments/assets/4585466c-a2ee-40ab-a793-8671730c4139)

Open a Notebook using Python 3.

![image](https://github.com/user-attachments/assets/d111c02f-fd22-4737-b8fb-c79935554c32)

You will be greeted with an ```Untitled.ipynb``` file. Go to the cell and enter

```
!pip install git+https://github.com/QuantAsylum/PyQa40x.git
```

as shown below. After entering, you can push the "Run this Cell" button, or just enter <kbd>SHIFT</kbd> + <kbd>ENTER</kbd>

![image](https://github.com/user-attachments/assets/73b86375-311b-4458-9952-d73052c5b27a)

This will install the library. 

Next, go to the PyQa40x directory you unzipped, and look in the ```LibTest``` subdirectory and open and copy the ```FirstPrinciples.py``` code. And then, paste that in the next cell in JupyterLab Desktop and run it. That should produce a graph similar to below:

![image](https://github.com/user-attachments/assets/625c3be2-026d-4e20-a454-8b88a906b0f9)

Next, copy the contents of the file ```PyQa40x_thdn.py``` into the next cell in JupyterLab Desktop and run that. At some point, you will run into errors telling you that libraries are missing. When you encounter an error message that a library is missing, you will need to add that library. For example, if you see the error on the line

```
import usb1  # pip install libusb1
```

the comment after the import error will indicate how to install the library. For example, ```pip install libusb1```. Just remember that if you want to run that command from a cell, you need to preface it with an exclaimation point such as ```!pip install libusb1```. 


