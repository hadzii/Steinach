#! /usr/bin/env python
# -*- coding: utf-8 -*-

def merge_single(nch,dstart,dend,dataDir):
  '''Merges traces of one channel to larger traces. Used for cross-correlation'''

  # here you load all the functions you need to use
  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream

  tr = []

  for k in range(dstart, dend, 1):
    fname = '%d' %(k)
    fileName = fname + ".dat" 
    st = readSEG2(dataDir + fileName)
    tr.append(st[nch-1])
 
  new_stream = Stream(traces=tr)
  new_stream.merge(method=1, fill_value='interpolate')
  return new_stream
  
  
  
def crossc(dstart,dend,ch1,ch2, Dir):
# here you load all the functions you need to use

  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream
  import numpy as np
  from obspy.signal.cross_correlation import xcorr
  from numpy import sign
  from obspy.signal.filter import lowpass
  from obspy.signal.filter import highpass
  from obspy.signal.filter import bandstop
  from obspy.signal.filter import bandpass



  ## loading the info for outfile-name
  #stream_start = readSEG2(Dir + str(dstart) + ".dat")
  #t_start = stream_start[ch1].stats.seg2.ACQUISITION_TIME
  #stream_end = readSEG2(Dir + str(dend) + ".dat")
  #t_end = stream_end[ch1].stats.seg2.ACQUISITION_TIME

  # initialization of the arrays and variables
  corr=0
  nerror = 0
    
  #TAPER
  taper_percentage=0.05
  taper= np.blackman(int(len(time_vector) * taper_percentage))
  taper_left, taper_right = np.array_split(taper,2)
  taper = np.concatenate([taper_left,np.ones(len(time_vector)-len(taper)),taper_right])
  
  for k in range(dstart, dend, 4):
    start = k
    end = k + 5 # only used to merge 5-1 = 4 files to one stream
    try:  
		 st1 = merge_single(ch1,start,end,Dir)
		 st2 = merge_single(ch2,start,end,Dir)
		 st1.detrend('linear')  
		 st2.detrend('linear') 
		 r = k-dstart

		 st1.filter('lowpass',freq = 24, zerophase=True, corners=8)
		 st1.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
		 st1.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
		 st2.filter('lowpass',freq = 24, zerophase=True, corners=8)
		 st2.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
		 st2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
		 
		 # 1-bit normalization
		 tr1 = sign(st1[0].data)
		 tr2 = sign(st2[0].data)    	
		 	
		 # cross-correlation
		 index, value, acorr = xcorr(tr1, tr2, 25000, full_xcorr=True)
		
		 # check sanity
		 if np.max(acorr)>1:
		   acorr = zeros(50001)   
		 
		 corr += acorr
		 print corr
    except:
      nerror += 1
      print "%d : ERROR" %(r)
  	
  corr = corr/np.max(np.abs(corr)) * taper

  return corr

  
  
  
  

####################################################################################################################
####################################################################################################################
# executive:

import matplotlib.pyplot as plt
import numpy as np
import csv
from obspy.seg2.seg2 import readSEG2
from obspy.signal.cross_correlation import xcorr


time=np.linspace(0,2.0, 4000)
time2=np.linspace(0,2.0, 1000)

##### PASSIVE

time_vector = np.linspace(-50,50,50001)
ch1 = 16
ch2 = 24
dstart= 1000001
dend= 1000113
dir_DEC = "/import/three-data/hadzii/STEINACH/STEINACH_longtime/"
dir_FEB = "/import/neptun-radler/STEINACH_feb/"

  
cross_DEC = crossc(dstart,dend,ch1,ch2,dir_DEC)
cross_FEB = crossc(dstart,dend,ch1,23,dir_FEB)


#### spectrum passive
tr = merge_single(24,1000001,1000020,dir_DEC) #### spectrum 0f 10 min --> sufficient, I think
spec_c = np.fft.rfft(tr[0].data)
fmax_c = 250  # sampling frequency divided by 2 --> Nyquist frequency
freq_c = np.linspace(0,fmax_c,len(spec_c))



##### ACTIVE

DEC_Dir= '/import/three-data/hadzii/STEINACH/Steinach/Steinach_ACTIVE/'
FEB_Dir= '/import/three-data/hadzii/STEINACH/Steinach/STEINACH_act_feb/'
DEC_fn='1005.dat'
FEB_fn='1000019.dat'

st_DEC = readSEG2(DEC_Dir + DEC_fn)
st_DEC.detrend('linear')

st_FEB = readSEG2(FEB_Dir + FEB_fn)
st_FEB.detrend('linear')
  
 
st_DEC.filter('lowpass',freq = 24, zerophase=True, corners=8)
st_DEC.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
st_DEC.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
st_FEB.filter('lowpass',freq = 24, zerophase=True, corners=8)
st_FEB.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
st_FEB.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)

st_DEC.normalize()
st_FEB.normalize()

#### spectrum active
spec=0
k=0
for ch in range(15,25):
  spec += np.fft.rfft(st_DEC[ch].data)
  k+=1
spec=spec/k
fmax = 1000  # sampling frequency divided by 2 --> Nyquist frequency
freq = np.linspace(0,fmax,len(spec))


#### 1bit normalization
#tr_DEC = np.sign(st_DEC[ch2-1].data)
#tr_FEB = np.sign(st_FEB[ch2-1].data)


#index1, value1, corr_DEC = xcorr(tr_DEC, cross_DEC[25000:29000], 1500, full_xcorr=True)
#index2, value2, corr_FEB = xcorr(tr_FEB, cross_FEB[25000:29000], 1500, full_xcorr=True)
#index3, value3, corr_g = xcorr(tr_DEC[0:4000], tr_FEB[0:4000], 1500, full_xcorr=True)

ax=plt.subplot(231)
ax.plot(time,st_DEC[ch2-1].data/np.max(np.abs(st_DEC[ch2-2].data)), linewidth=2)
ax.set_title('passive data ccf causal part (red) and active data (blue) for DEC20 2011', fontsize=12, fontweight='bold')
ax.set_xlabel('time [s]')
ax.set_ylabel('normalized amplitude')
axx=ax.twiny()
axx.plot(cross_DEC[25000:26000]/np.max(np.abs(cross_DEC[25000:26000])), color='r', linewidth=2) #### starts at 25000 for causal part of the ccf
axx.set_axis_off()


ax2=plt.subplot(232)
ax2.fill_between(freq,0,np.abs(spec))
#ax2.set_xticks(np.linspace(0,1000,44))
ax2.set_ylabel('amplitude')
ax2.set_xlabel('frequency [Hz]')
ax2.set_title('Stacked spectrum of active data', fontsize=12, fontweight='bold')
ax2.set_xlim(0,250)


ax3=plt.subplot(234)
ax3.plot(time,st_FEB[ch2-1].data[1000:5000]/np.max(np.abs(st_FEB[ch2-1].data[1000:5000])), linewidth=2)
ax3.set_title('passive data ccf causal part (red) and active data (blue) for FEB1 2011', fontsize=12, fontweight='bold')
ax3.set_xlabel('time [s]')
ax3.set_ylabel('normalized amplitude')
axx3=ax3.twiny()
axx3.plot(cross_FEB[25000:26000]/np.max(np.abs(cross_FEB[25000:26000])), color='r', linewidth=2) #### starts at 25000 for causal part of the ccf
axx3.set_axis_off()


ax4=plt.subplot(235)
ax4.fill_between(freq_c, 0, np.abs(spec_c))
ax4.set_xticks(np.linspace(0,250,11))
ax4.set_ylabel('amplitude')
ax4.set_xlabel('frequency [Hz]')
ax4.set_title('Stacked spectrum of passive data', fontsize=12, fontweight='bold')
ax4.set_xlim(0,250)

ax5=plt.subplot(233)
ax5.plot(time2,cross_FEB[25000:26000]/np.max(np.abs(cross_FEB[25000:26000])), color='b', linewidth=2) #### starts at 25000 for causal part of the ccf
ax5.plot(time2,cross_DEC[25000:26000]/np.max(np.abs(cross_DEC[25000:26000])), color='r', linewidth=2)
ax5.set_title('passive data ccf causal part for DEC20 (red) and FEB1 (blue)', fontsize=12, fontweight='bold')
ax5.set_xlabel('time [s]')
ax5.set_ylabel('normalized amplitude')


ax6=plt.subplot(236)
ax6.plot(time,st_FEB[ch2-1].data[1000:5000]/np.max(np.abs(st_FEB[ch2-1].data[1000:5000])), linewidth=2, color='b') #### the 1000:5000 is because the FEB data is delayed by 1000 samples compared to the DEC data and is longer also
ax6.plot(time,st_DEC[ch2-1].data/np.max(np.abs(st_DEC[ch2-1].data)), linewidth=2, color='r')
ax6.set_title('active data for FEB1 (blue) and DEC20 (red)', fontsize=12, fontweight='bold')
ax6.set_xlabel('time [s]')
ax6.set_ylabel('normalized amplitude')

plt.show()

