PyQa40x allows you to run Python code to communicate with the QA40x hardware. No intermediate program is needed, as the PyQa40x lib understands how to use the calibration data stored inside the QA40x hardware.

The library is very much in flux, and support might be sporadic. The PyQa40x lib is one of several ways to [use the QA40x hardware](https://github.com/QuantAsylum/QA40x/wiki/QA40x-API). There are now 5 ways to interact with your QA40x hardware:

1) Using the QA40x application. This is the most-supported path. An addition to making a host of basic audio measurements, the QA40x app can also do swept plots, which is useful for plotting thinks like THD versus Level, for example. The application is located [here](https://github.com/QuantAsylum/QA40x/releases)
2) Tractor. Tractor allows you create scripts to drive automated testing. You can establish pass/fail limits, allowing an operating with limited skills to test products in a production environment. Tractor works in conjuction with the QA40x application to "drive" your tests. The application is located [here](https://github.com/QuantAsylum/Tractor/releases/tag/v1.101)
3) REST. REST is a common way of interacting with other entities on the internet, and every modern language provides support for implementing REST. The QA40x application handles the REST communication, and allows you to use GET/PUT/POST to control the measurements. A test program in C# is located [here](https://github.com/QuantAsylum/QA402_REST_TEST). More info on the Wiki is located [here](https://github.com/QuantAsylum/QA40x/wiki/QA40x-API)
4) Bare Metal. If you'd like to see how to communicate with the QA40x hardware directly with c#, you could look at the repo located [here](https://github.com/QuantAsylum/QA40x_BareMetal)
5) ASIO. An ASIO driver for the QA403/QA402/QA401 is available [here](https://github.com/dechamps/ASIO401), allowing you to use the QA40x hardware with common audio-processing applications.
6) PyQa40x. This library is an attempt to streamline a lot of common processing operations, while handling some of the more difficult topics such as normalization and windowing. The aim here is to facilitate an environment to digital signal processing development using absolute values instead of relative values. What this means is if you specify a 1Vrms sine way, the output of of the QA40x hardware will be 1Vrms.

The rest of this document focuses on 

If you are looking for a simple starting point, 

To get started, you can install the library in the Jupyter Lab environment by typing the following into a Juptyer Lab cell and the running the command:

```
!pip install git+https://github.com/QuantAsylum/PyQa40x.git
```

After the installation has finished, there are examples in the LibTest directory you can run.


