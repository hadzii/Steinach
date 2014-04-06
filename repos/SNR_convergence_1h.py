#! /usr/bin/env python
# -*- coding: utf-8 -*-

def merge_single(nch,dstart,dend):
  '''Merges traces of one channel to larger traces. Used for cross-correlation'''

  # here you load all the functions you need to use
  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream


  # directories:
  dataDir2 = "/import/neptun-radler/STEINACH_feb/"
  dataDir = "/import/three-data/hadzii/STEINACH/STEINACH_longtime/"


  tr = []

  for k in range(dstart, dend, 1):
    fname = '%d' %(k)
    fileName = fname + ".dat" 
    st = readSEG2(dataDir + fileName)
    tr.append(st[nch-1])


  
  new_stream = Stream(traces=tr)
  new_stream.merge(method=1, fill_value='interpolate')


  return new_stream
  
  
def est_SNR_1h(dstart,dend,ch1,ch2):
# here you load all the functions you need to use

  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream
  import numpy as np
  from obspy.signal.cross_correlation import xcorr
  from numpy import sign

  dataDir = "/import/three-data/hadzii/STEINACH/STEINACH_longtime/"
  SNR=[]

	# loading the info for outfile-name
  stream_start = readSEG2(dataDir + str(dstart) + ".dat")
  t_start = stream_start[ch1].stats.seg2.ACQUISITION_TIME
  stream_end = readSEG2(dataDir + str(dend) + ".dat")
  t_end = stream_end[ch1].stats.seg2.ACQUISITION_TIME

  
  for k in range(dstart, dend, 1):
   st1 = merge_single(ch1,dstart,k+1)
	 st2 = merge_single(ch2,dstart,k+1)

	 st1.detrend('linear')  
	 st2.detrend('linear') 

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
   SNR_ges=[]
   for isnrb3 in range(40000,49001,500):  # steps of half a windowlength
    endwb3=isnrb3 + 1000  # 1s windows
    SNR_ges.append(np.max(np.abs(corr))/np.std(corr[isnrb3:endwb3]))

   SNR.append(np.max(SNR_ges))
	   
  return SNR



####################################################################################################################
####################################################################################################################
# executive:

import matplotlib.pyplot as plt
import numpy as np

time_vector = np.linspace(-50.0,50.0,50001)
ch1 = 6
ch2 = 16

start=1000001
end=1000113
	

    # INITIALIZATIONS

    
SNR = np.zeros((end-start,1))

SNR = est_SNR_1h(start,end,ch1,ch2)
print SNR


    
    

	 
	  

	  
	  