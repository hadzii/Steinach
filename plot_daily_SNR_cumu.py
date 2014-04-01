#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.axes import Axes


time_vector = np.linspace(-10.0,10.0,10001)
ch1 = 6
ch2 = 20
day = 7
k=-1
SNR_type = "SNR_calm_b2"
mode = "calm"
counter = "nbeat"
sum=0


ind = np.arange(0,24)


SNR_vec = []
SNR_vec2 = []
SNR_cumu = 0
h = np.arange(0,24,1)
f1=0
path = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/*" + mode + "_" +"2-8Hz"+  "_CH" + str(ch2) + ".npy"






ax1 = plt.subplot(131)
ax1.set_title('Cumulative Cross-Correlations converging to average')
ax1.set_ylabel('Hour of day')

ax2 = plt.subplot(133)
ax2.set_title('Number of Beat files')
ax2.grid()

ax3 = plt.subplot(132)
ax3.set_title('Cumulative Signal-to-Noise Ratio')
ax3.grid()



for fname in glob.glob(path):
     SNR_f2 = []
     SNR_f1 = []
     k+=1
     f=np.load(fname)
     f1 +=f
     ax1.plot(time_vector,f1/np.max(np.abs(f1))+k) # normalized plots that cumulatively converge to the 24h average
     # cumulative SNR
     for isnrb1 in range(7500,9001,500):  # steps of half a windowlength
       endwb1=isnrb1 + 250  # 5s window
       SNR_f1.append(np.max(np.abs(f1))/np.std(f1[isnrb1:endwb1]))
       SNR_f2.append(np.max(np.abs(f))/np.std(f[isnrb1:endwb1]))
     SNR_cumu = max(SNR_f1)
     SNR_cumu2 = max(SNR_f2)

     SNR_vec.append(SNR_cumu) 
     SNR_vec2.append(SNR_cumu2) 


count = np.load("/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) +"/" + counter + ".npy")

#SNR2=[]
#for k in range(0,24):
#	SNR2.append(int(SNR[k]))
#print SNR2

p1, = ax3.plot(SNR_vec,h,linewidth=3,color='g')
p2, = ax3.plot(SNR_vec2,h, color = 'r')
p3, = ax2.plot(count,h,color= 'b', linewidth=2)
#ax3.bar(ind, SNR2, width=0.8, orientation='horizontal')

ax1.set_ylim((-1,24))
ax2.set_ylim((-1,24))
ax3.set_ylim((-1,24))

ax3.legend([p1,p2], ['cumulative SNR', 'SNR'])
ax2.legend([p3], [counter])



plt.show()
