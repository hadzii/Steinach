#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


time_vector = np.linspace(-50.0,50.0,50001)
ch1 = 6
ch2 = 7
day = 8


path = "/home/jsalvermoser/Desktop/VERGLECIH_JAN8_CH6_CH7/JAN8_detrended_simple_before+after/"
path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/"

fname = path2 + "04:11:36-05:13:06CH6_xcorr128s_calm_0-2Hz_CH7.npy"
f=np.load(fname)
print f
print len(f)
plt.plot(time_vector,f/np.max(np.abs(f)))

plt.show()