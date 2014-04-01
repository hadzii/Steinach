#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import glob



time_vector = np.linspace(-10.0,10.0,10001)
ch1 = 6
ch2 = 20
day = 7
k=15
mode  = "beat"
SNR_type = "SNR_beat_b2"
sum1 = 0
band="2-8Hz"       

ax = plt.subplot(111)

for ch2 in range(16,48):
  sum1 = 0
  try:
    path = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/*" + mode + "_" +band+  "_CH" + str(ch2) + ".npy"
    for fname in glob.glob(path):
     f=np.load(fname)
     sum1 += f 
    k+=1
    ax.plot(sum1/np.max(np.abs(sum1))+k)
  except:
    print "no such directory"


plt.show()