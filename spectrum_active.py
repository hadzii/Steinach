#! /usr/bin/env python
# -*- coding: utf-8 -*-

from obspy.seg2.seg2 import readSEG2
dataDir = "/home/jsalvermoser/Desktop/STEINACH_act_feb/"
path = "/home/jsalvermoser/Desktop/STEINACH_act_feb/*.dat"
import matplotlib.pyplot as plt
import numpy as np
import glob
spec = 0
k=0
for fname in glob.glob(path):
  st=readSEG2(fname)
  spec += np.fft.rfft(st[5].data)
  k+=1
  print fname
print k
spec_sum = spec/k

#fileName = '1000006.dat'
#st = readSEG2(dataDir + fileName)
#data = st[5].data
#spec1 = np.fft.rfft(data)
fmax = 1000  # sampling frequency divided by 2 --> Nyquist frequency
freq = np.linspace(0,fmax,len(spec))


# lowpass-filter the crossc_beat correlation function 




#tr2 = tr.filter('lowpass', freq = 24, zerophase=True, corners=8)
#tr2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
#spec2 = np.fft.rfft(tr2.data)


ax = plt.subplot(111)
ax.plot(freq,np.abs(spec)) # take absolute value as the spectrum is complex valued
ax.set_xticks(np.linspace(0,1000,11))
ax.set_ylabel('amplitude')
ax.set_xlabel('frequency [Hz]')
ax.set_title(str(st[0].stats.seg2.ACQUISITION_TIME))
#plt.subplot(212)
#plt.plot(freq,spec2)

plt.show()