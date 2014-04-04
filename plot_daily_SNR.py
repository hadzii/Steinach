#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.axes import Axes


time_vector = np.linspace(-50.0,50.0,50001)
ch1 = 6
ch2 = 7
day = 8
k=-1
mode  = "beat"
counter = "nbeat"
SNR_type = "SNR_beat_b3"
band = "8-24Hz"
sum=0
decaytime = []
idec=0

ind = np.arange(-0.1,23.9,1)
h = np.arange(0,24,1)

path = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/*" + mode + "_" + band +  "_CH" + str(ch2) + ".npy"

ax1 = plt.subplot(131)
ax1.set_title('Hourly Cross-Correlations')

ax1.set_ylabel('Hour of day')

ax2 = plt.subplot(132)
ax2.set_title('Number of Beat files')
ax2.grid()

ax3 = plt.subplot(133)
ax3.set_title('Signal-to-Noise Ratio')
ax3.grid()



for fname in glob.glob(path):
  k+=1
  f=np.load(fname)
  ax1.plot(time_vector,f/np.max(np.abs(f))+k)
  if np.max(f)>0:
    sum += f #/np.max(np.abs(f))

     

ax1.plot(time_vector,sum/np.max(np.abs(sum))-2,linewidth = 3)


count = np.load("/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) +"/" + counter + ".npy")
SNR = np.load("/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) +"/" + SNR_type + ".npy")
SNR2=[]

for k in range(0,24):
  if SNR[k]< 10000000:
    SNR2.append(int(SNR[k]))
  else:
    SNR2.append(0)
#ax2.plot(count,h,linewidth=3,color='r')
#ax4.plot(decaytime,h,'bo')

ax2.barh(ind,width=count,height = 0.2,color='r') 
ax3.barh(ind,width=SNR2,height = 0.2, color='g')

ax1.set_ylim(-3,25)
ax2.set_ylim(-3,25)
ax3.set_ylim(-3,25)




plt.show()
