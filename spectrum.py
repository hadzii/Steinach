#! /usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import read
from merge_single import merge_single
import matplotlib.pyplot as plt
import numpy as np

tr = merge_single(6,1000676,1001350)
spec1 = np.fft.rfft(tr.data)
fmax = 250  # sampling frequency divided by 2 --> Nyquist frequency
freq = np.linspace(0,fmax,len(spec1))


# lowpass-filter the crossc_beat correlation function 




tr2 = tr.filter('lowpass', freq = 24, zerophase=True, corners=8)
tr2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
spec2 = np.fft.rfft(tr2.data)


ax = plt.subplot(111)
ax.plot(freq,np.abs(spec1)) # take absolute value as the spectrum is complex valued
ax.set_xticks(np.linspace(0,250,126))
ax.set_ylabel('amplitude')
ax.set_xlabel('frequency [Hz]')
ax.set_title(str(tr.stats.starttime) + '    -    '  + str(tr.stats.endtime))
#plt.subplot(212)
#plt.plot(freq,spec2)

plt.show()