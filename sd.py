

import matplotlib.pyplot as plt
from merge_single import merge_single
from numpy import sign
import numpy as np
from obspy.signal.cross_correlation import xcorr
corr=0

ax = plt.subplot(111)
time_vector = np.linspace(-50.0,50.0,50001)
for k in range(1048728,1048840,4):
 end=k+4
 print end
 tr1=merge_single(6,k,end)
 tr2=merge_single(7,k,end)
 tr1.detrend('linear')
 tr2.detrend('linear')
 tr1.filter('bandpass', freqmin=0.1, freqmax=2, corners=2, zerophase=True)
 tr2.filter('bandpass', freqmin=0.1, freqmax=2, corners=2, zerophase=True)
 tr1=sign(tr1.data)
 tr2=sign(tr2.data)

 index,value,acorr = xcorr(tr1, tr2, 25000, full_xcorr=True)
 print acorr
 ax.plot(time_vector,acorr/np.max(acorr) +k-1048728)
 corr+=acorr
ax.plot(time_vector,corr/np.max(corr)-4)
plt.show()