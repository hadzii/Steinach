#! /usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import read
from merge_single import merge_single
import matplotlib.pyplot as plt
import numpy as np

tr = merge_single(6,1001001,1001113)
ax=plt.subplot(121)
ax2=plt.subplot(122)

time=np.linspace(0,len(tr)/500,len(tr))

#### Spectrogram
tr.spectrogram(wlen=1, per_lap=0.5, mult=1, dbscale=True, title='Spectrogram unfiltered for one hour', axes=ax2)
ax2.set_title('Spectrogram unfiltered for one hour', fontsize=15, fontweight='bold')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('frequency [Hz]')

#### Spectrum

spec1 = np.fft.rfft(tr.data)
fmax = 250  # sampling frequency divided by 2 --> Nyquist frequency
freq = np.linspace(0,fmax,len(spec1))
# lowpass-filter the crossc_beat correlation function 
tr2 = tr.filter('lowpass', freq = 24, zerophase=True, corners=8)
tr2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
spec2 = np.fft.rfft(tr2.data)
ax.plot(freq,np.abs(spec1), color='DodgerBlue',label='unfiltered') # take absolute value as the spectrum is complex valued
ax.plot(freq,np.abs(spec2), color='Darkred',label='filtered')
ax.set_xticks(np.linspace(0,250,26))
ax.set_ylabel('amplitude')
ax.set_xlabel('frequency [Hz]')
ax.set_yscale('log')
ax.legend()



plt.show()


