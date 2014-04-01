

import matplotlib.pyplot as plt
from merge_single import merge_single
from numpy import sign
import numpy as np
from obspy.signal.cross_correlation import xcorr
from obspy.signal.filter import lowpass
from obspy.signal.filter import highpass
from obspy.signal.filter import bandstop
from obspy.signal.filter import bandpass
corr=0

time_vector = np.linspace(-50.0,50.0,50001)
taper_percentage=0.2
taper= np.blackman(int(len(time_vector) * taper_percentage))
taper_left, taper_right = np.array_split(taper,2)
taper = np.concatenate([taper_left,np.ones(len(time_vector)-len(taper)),taper_right])

ax = plt.subplot(111)

for k in range(1048728,1048840,4):
 end=k+4
 print end
 tr1=merge_single(6,k,end)
 tr2=merge_single(7,k,end)
 tr1.detrend('linear')
 tr2.detrend('linear')
 tr1.filter('lowpass',freq = 24, zerophase=True, corners=8)
 tr1.filter('highpass', freq= 0.05, zerophase=True, corners=2)
 tr1.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
 tr2.filter('lowpass',freq = 24, zerophase=True, corners=8)
 tr2.filter('highpass', freq= 0.05, zerophase=True, corners=2)
 tr2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
 tr1=sign(tr1.data)
 tr2=sign(tr2.data)
 
 
 index,value,acorr = xcorr(tr1, tr2, 25000, full_xcorr=True)
 acorr = acorr*taper
 acorr = highpass(acorr, freq=0.1, corners=4, zerophase=True, df=500.)
 acorr = lowpass(acorr, freq=2, corners=4, zerophase=True, df=500.)
 ax.plot(time_vector,acorr/np.max(acorr) +k-1048728)
 corr+=acorr
ax.plot(time_vector,corr/np.max(corr)-4)
plt.show()