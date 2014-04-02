#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.axes import Axes


time_vector = np.linspace(-50.0,50.0,50001)
ch1 = 6
ch2 = 25
day = 8
sec=128
k=-1
mode  = "beat"
counter = "nbeat"
SNR_type = "SNR_beat_b2"
band = "8-24Hz"
sum=0
decaytime1 = []
decaytime2 = []
max_index=0

ind = np.arange(-0.1,23.9,1)
h = np.arange(0,24,1)

path = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/*" + mode + "_" + band +  "_CH" + str(ch2) + ".npy"

# SUB-PLOT1
ax1 = plt.subplot(141)


for fname in glob.glob(path):
     k+=1
     ampmax = 0
     amp = 0
     amplitudes = []
     notdecayed1 = []
     notdecayed2 = []
     f=np.load(fname)
     if np.max(f)>10.0:
       f = np.zeros(50001)
     
     # this is done to calculate the decaytime of a cross-correlation function
     # first check which frequency band is used and assign a sliding window length
     if band == "0-2Hz":
     	window=500
     elif band == "2-8Hz":
     	window=250
     else:
     	window=125
     dim = ((50000-window)-(25000-window/2.))/(window/2) # dimension for counter "l"
     for idecay in range(int(25000-window/2),50000-window, int(window/2.)):
        startwindow = idecay
     	endwindow = idecay + window
     	amp=0
     	for pot in range(startwindow,endwindow):
     		amp += np.abs(f[pot])
     	amplitudes.append(amp)
     maxm = np.max(amplitudes)	
     for index, item in enumerate(amplitudes):
	if item == maxm:
	  max_index = index
     if amplitudes[0]!=0:
      for l in range(max_index,int(dim)): # need to start from here as the maximum is not always in the center of the function
  	  if amplitudes[l]> 0.5* np.max(amplitudes):
	    notdecayed2.append(l*125/500.)      
	  elif amplitudes[l]> 0.25* np.max(amplitudes):
	    notdecayed1.append(l*125/500.) # decaytime in seconds

      decaytime1.append(np.max(notdecayed1))
      decaytime2.append(np.max(notdecayed2))
     else:
      decaytime.append(0)    
       
     
     ax1.plot(time_vector,f/np.max(np.abs(f))+k)
     if np.max(f)>0:
     	sum += f/np.max(np.abs(f))

     
ax1.plot(time_vector,sum/np.max(np.abs(sum))-2,linewidth = 3)


count = np.load("/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) +"/" + counter + ".npy")
SNR = np.load("/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) +"/" + SNR_type + ".npy")
SNR2=[]

for k in range(0,24):
  if SNR[k]< 1000000:
    SNR2.append(int(SNR[k]))
  else:
    SNR2.append(0)
#ax2.plot(count,h,linewidth=3,color='r')
#ax4.plot(decaytime,h,'bo')

# SUB-PLOTS 2-4

plt.suptitle("Daily plot of hourly Cross-Correlation functions using " + str(sec)+ "s-files" ,fontsize=22)

ax1.set_ylim(-3,25)
ax1.set_xlabel('time [s]')
ax1.set_title('Hourly Cross-Correlations', fontsize=18)
ax1.set_ylabel('Hour of day')

ax2 = plt.subplot(142)
ax2.grid()
ax2.barh(ind,width=count,height = 0.2,color='r') 
ax2.set_ylim(-3,25)
str1='Number of '
str2= 'merged ' ### ATTENTION: CHANGE IF NO MERGED FILES
str3=' files'
caption = ''.join((str1,str2,mode,str3))
ax2.set_xlabel(caption)
title=''.join((str1,mode,str3))
ax2.set_title(title,color="r", fontsize=18)

ax3 = plt.subplot(143)
ax3.set_title('Signal-to-Noise Ratio', color="g", fontsize=18)
ax3.grid()
ax3.barh(ind,width=SNR2,height = 0.2, color='g')
ax3.set_ylim(-3,25)
ax3.set_xlabel('SNR')

ax4 = plt.subplot(144)
ax4.set_title('Decaytime',color="b", fontsize=18)
ax4.grid()
b1 = ax4.barh(ind,width=decaytime1,height = 0.2, color='b',label='75% decayed')
b2 = ax4.barh(ind,width=decaytime2,height = 0.2, color='y', label='50% percent decayed')
ax4.set_ylim(-3,25)
ax4.set_xlabel('decaytime [s]')
ax4.legend()

plt.show()
