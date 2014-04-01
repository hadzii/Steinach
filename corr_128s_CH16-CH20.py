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
  
  
def crossc(dstart,dend,ch1,ch2,day):
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
  dataDir = "/import/three-data/hadzii/STEINACH/STEINACH_longtime/"
  outdir = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN" + str(day) + "/"


	# loading the info for outfile-name
  stream_start = readSEG2(dataDir + str(dstart) + ".dat")
  t_start = stream_start[ch1].stats.seg2.ACQUISITION_TIME
  stream_end = readSEG2(dataDir + str(dend) + ".dat")
  t_end = stream_end[ch1].stats.seg2.ACQUISITION_TIME

	# initialization of the arrays and variables
  TR = []
  rms = []
  sq = []
  ncalm = 1
  nbeat  = 1
  corr128_calm = 0
  corr128_beat = 0
  nerror = 0
  mu1c=0
  mu2c=0
  mu3c=0
  mu1b=0
  mu2b=0
  mu3b=0
  var1c=0
  var2c=0
  var3c=0
  var1b=0
  var2b=0
  var3b=0
  SNR_calm_b1=[]
  SNR_calm_b2=[]
  SNR_calm_b3=[]
  SNR_beat_b1=[]
  SNR_beat_b2=[]
  SNR_beat_b3=[]
  
  
  #TAPER
  taper_percentage=0.2
  taper= np.blackman(int(len(time_vector) * taper_percentage))
  taper_left, taper_right = np.array_split(taper,2)
  taper = np.concatenate([taper_left,np.ones(len(time_vector)-len(taper)),taper_right])
  
  for j in range(0, dend-dstart):
    sq.append([])



  for k in range(dstart, dend, 4):
    start = k
    end = k + 5 # only used to merge 5-1 = 4 files to one stream
    try:  
		 st1 = merge_single(ch1,start,end)
		 st2 = merge_single(ch2,start,end)
		 st1.detrend('linear')  
		 st2.detrend('linear') 
		 # calculate squares for rms
		 r = k-dstart
		 sq[r] = 0
		 for h in range(0,64000):
		   sq[r] += (st1[0].data[h])**2   
		     # lowpass-filter the crossc_beat correlation function 
		 st1.filter('lowpass',freq = 24, zerophase=True, corners=8)
		 st1.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
		 st1.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
		 st2.filter('lowpass',freq = 24, zerophase=True, corners=8)
		 st2.filter('highpass', freq= 0.05, zerophase=True, corners=2) #had to be reduced from 0.1Hz
		 st2.filter('bandstop', freqmin=8, freqmax=14, corners=4, zerophase=True)
		 
		 	# sometimes channels seem to fail, so I put this to prevent crashing of the program
		 
		 
		 	# 1-bit normalization
		 tr1 = sign(st1[0].data)
		 tr2 = sign(st2[0].data)    	
		 	
		 # cross-correlation
		 index, value, acorr = xcorr(tr1, tr2, 25000, full_xcorr=True)
		
		 print sq[r]
		 	
		 # sort the 128sec files into calm and beat:
		 # the value was chosen after observing calm files
		 	
		 if sq[r] < 1000000000000:
		     corr128_calm += acorr
		     ncalm += 1.
		 else:
		     corr128_beat += acorr
		     nbeat += 1.
		 print ncalm,  nbeat  # just to check if calm or noisy
    except:
      nerror += 1
      print "%d : ERROR" %(r)
	 	
	 	
  if ncalm<8:
  	corr128_calm = np.zeros(50001)
  	
  # normalization	 	
  else:
  	corr128_calm = (corr128_calm/ncalm) * taper
  	
  corr128_beat = (corr128_beat/nbeat) * taper


  # filter again and divide into 3 bands which can be investigated separately
  
  corr128_calm_band1 = highpass(corr128_calm, freq=0.1, corners=4, zerophase=True, df=500.)
  corr128_calm_band1 = lowpass(corr128_calm_band1, freq=2, corners=4, zerophase=True, df=500.)
  corr128_calm_band2 = bandpass(corr128_calm, freqmin=2, freqmax=8, df=500., corners=4, zerophase=True)
  corr128_calm_band3 = bandpass(corr128_calm, freqmin=8, freqmax=24, df=500., corners=4, zerophase=True)
  corr128_beat_band1 = bandpass(corr128_beat, freqmin=0.5, freqmax=2, df=500., corners=2, zerophase=True)
  corr128_beat_band2 = bandpass(corr128_beat, freqmin=2, freqmax=8, df=500., corners=4, zerophase=True)
  corr128_beat_band3 = bandpass(corr128_beat, freqmin=8, freqmax=24, df=500., corners=4, zerophase=True)
  
  # SNR (Signal-to-Noise Ratio):print 222222
  # for the signal-to-noise ratio one divides the maximum of the signal by the
  # variance of a late window (noise). As we don't know which window has the
  # lowest signal fraction, we loop over some windows. We need windows of 
  # different lengths for the different bands as different frequencies are 
  # contained. For every band the minimum-frequency fmin is chosen (e.g. 4Hz), then
  # the time for one cyle is 1/fc (e.g. 0.25s) and as we take windows of 3-4 
  # cycles we choose a window length of 4*0.25s = 1s
  
  ## CALM + BEAT
  for isnrb1 in range(45000,50000,2500):  # steps of half a windowlength
    endwb1=isnrb1 + 2500  # 5s window
    SNR_calm_b1.append(np.max(np.abs(corr128_calm_band1))/np.std(corr128_calm_band1[isnrb1:endwb1]))
    SNR_beat_b1.append(np.max(np.abs(corr128_beat_band1))/np.std(corr128_beat_band1[isnrb1:endwb1]))
  SNR_calm_b1 = max(SNR_calm_b1)
  SNR_beat_b1 = max(SNR_beat_b1)
  
  for isnrb2 in range(45000,49001,500):  # steps of half a windowlength
    endwb2=isnrb2 + 1000  # 2s windows
    SNR_calm_b2.append(np.max(np.abs(corr128_calm_band2))/np.std(corr128_calm_band2[isnrb2:endwb2]))
    SNR_beat_b2.append(np.max(np.abs(corr128_beat_band2))/np.std(corr128_beat_band2[isnrb2:endwb2]))
  SNR_beat_b2 = max(SNR_beat_b2)
  SNR_calm_b2 = max(SNR_calm_b2)
  
  for isnrb3 in range(45000,49751,125):  # steps of half a windowlength
    endwb3=isnrb3 + 250  # 0.5s windows
    SNR_calm_b3.append(np.max(np.abs(corr128_calm_band3))/np.std(corr128_calm_band3[isnrb3:endwb3]))
    SNR_beat_b3.append(np.max(np.abs(corr128_beat_band3))/np.std(corr128_beat_band3[isnrb3:endwb3]))
  SNR_beat_b3 = max(SNR_beat_b3)
  SNR_calm_b3 = max(SNR_calm_b3)
  
  if ncalm<10:
  	SNR_calm_b1 = 0
  	SNR_calm_b2 = 0
  	SNR_calm_b3 = 0

  print SNR_calm_b1, SNR_calm_b2, SNR_calm_b3
  print SNR_beat_b1, SNR_beat_b2, SNR_beat_b3
    	
  # RMS for histogram and sifting:
  #for s in range(0,dend-dstart):
  #  rms.append((sq[s]/16000)**(0.5))

  # save into files:
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_beat_0-2Hz" + "_" + "CH" + str(ch2), corr128_beat_band1)
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_beat_2-8Hz" + "_" + "CH" + str(ch2), corr128_beat_band2)
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_beat_8-24Hz" + "_" + "CH" + str(ch2), corr128_beat_band3)
  
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_calm_0-2Hz" + "_" + "CH" + str(ch2), corr128_calm_band1)
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_calm_2-8Hz" + "_" + "CH" + str(ch2), corr128_calm_band2)
  np.save(outdir + t_start + "-" +  t_end + "CH" + str(ch1) + "_" +"xcorr128s_calm_8-24Hz" + "_" + "CH" + str(ch2), corr128_calm_band3) 

  # np.save(outdir + "JAN_"+"CH" + str(ch1) + "_" +"RMS" + "_" + "CH" + str(ch2) + str(dstart) + "-" + str(dend), rms)
  
  
  return corr128_beat_band1,corr128_beat_band2,corr128_beat_band3, corr128_calm_band1,corr128_calm_band2,corr128_calm_band3, ncalm, nbeat, SNR_beat_b1, SNR_beat_b2, SNR_beat_b3, SNR_calm_b1, SNR_calm_b2, SNR_calm_b3

  # time-axis for plotting:
  # dt = st[0].stats.starttime.timestamp
  # t = np.linspace(st[0].stats.starttime.timestamp - dt,
  #  st[0].stats.endtime.timestamp -dt, st[0].stats.npts)
  
  
  
  

####################################################################################################################
####################################################################################################################
# executive:

import matplotlib.pyplot as plt
import numpy as np

time_vector = np.linspace(-50.0,50.0,50001)
ch1 = 6
dend = []


	
for ic in range(16,22,2):
  ch2 = ic
  path = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + "JAN"
  
  first = 1029828  # here begins JAN 8th
  
  # loop over the days that are to be investigated:
  # 2700 is #files for one day
  
  for iday in range(7,8):
    outDir = path + str(iday+1) + "/"
    
    # INITIALIZATIONS
    corr_beat_band1 = np.zeros((24,50001)) # produces 2d array
    corr_beat_band2 = np.zeros((24,50001))
    corr_beat_band3 = np.zeros((24,50001))
    nbeat = np.zeros((24,1))  # produces 1d array
    
    corr_calm_band1 = np.zeros((24,50001))
    corr_calm_band2 = np.zeros((24,50001))
    corr_calm_band3 = np.zeros((24,50001))
    ncalm = np.zeros((24,1))
    
    SNR_beat_b1 = np.zeros((24,1))
    SNR_beat_b2 = np.zeros((24,1))
    SNR_beat_b3 = np.zeros((24,1))
    SNR_calm_b1 = np.zeros((24,1))
    SNR_calm_b2 = np.zeros((24,1))
    SNR_calm_b3 = np.zeros((24,1))
    
    sum_beat_band1 = 0
    sum_beat_band2 = 0
    sum_beat_band3 = 0
    sum_calm_band1 = 0 
    sum_calm_band2 = 0
    sum_calm_band3 = 0
    
    
    
    
    dend.append(first + iday* 2700)
    dend.append(int(first + 112.5 + 0.5 + iday*2700))
    dend.append(int(first + 2*112.5 + iday*2700))
    dend.append(int(first + 3*112.5 - 0.5 + iday*2700))
    dend.append(int(first + 4*112.5 + iday*2700))
    dend.append(int(first + 5*112.5 + 0.5 + iday*2700)) 
    dend.append(int(first + 6*112.5 + iday*2700))
    dend.append(int(first + 7*112.5 -0.5 + iday*2700))
    dend.append(int(first + 8*112.5 + iday*2700))
    dend.append(int(first + 9*112.5 + 0.5 + iday*2700)) 
    dend.append(int(first + 10*112.5 + iday*2700))
    dend.append(int(first + 11*112.5 -0.5 + iday*2700))
    dend.append(int(first + 12*112.5 + iday*2700))
    dend.append(int(first + 13*112.5 + 0.5 + iday*2700))
    dend.append(int(first + 14*112.5 + iday*2700))
    dend.append(int(first + 15*112.5 -0.5 + iday*2700))
    dend.append(int(first + 16*112.5 + iday*2700))
    dend.append(int(first + 17*112.5 + 0.5 + iday*2700)) 
    dend.append(int(first + 18*112.5 + iday*2700))
    dend.append(int(first + 19*112.5 -0.5 + iday*2700))
    dend.append(int(first + 20*112.5 + iday*2700))
    dend.append(int(first + 21*112.5 + 0.5 + iday*2700)) 
    dend.append(int(first + 22*112.5 + iday*2700))
    dend.append(int(first + 23*112.5 -0.5 + iday*2700))
    dend.append(int(first + 24*112.5 + iday*2700))
    
    # evaluate the function above to get the data-arrays
    # EVALUATION
    for ih in range(0,24):  # ih is the counter for hours
	  corr_beat_band1[ih,:],corr_beat_band2[ih,:],corr_beat_band3[ih,:],\
	  corr_calm_band1[ih,:],corr_calm_band2[ih,:],corr_calm_band3[ih,:],\
	  ncalm[ih], nbeat[ih], SNR_beat_b1[ih], SNR_beat_b2[ih],\
	  SNR_beat_b3[ih], SNR_calm_b1[ih], SNR_calm_b2[ih], SNR_calm_b3[ih] =\
	  crossc(dend[ih],dend[ih+1],ch1,ch2,iday+1)

    try:
		np.save(outDir + 'nbeat', nbeat)
		np.save(outDir + 'ncalm', ncalm)	
		np.save(outDir + 'SNR_beat_b1', SNR_beat_b1)
		np.save(outDir + 'SNR_beat_b2', SNR_beat_b2)
		np.save(outDir + 'SNR_beat_b3', SNR_beat_b3)
		np.save(outDir + 'SNR_calm_b1', SNR_calm_b1)
		np.save(outDir + 'SNR_calm_b2', SNR_calm_b2)
		np.save(outDir + 'SNR_calm_b3', SNR_calm_b3)

	
    except:
	  print 'Could not save nbeat/ncalm'
    
    for isum in range(0,24):
	  sum_beat_band1 += corr_beat_band1[isum]
	  sum_beat_band2 += corr_beat_band2[isum]
	  sum_beat_band3 += corr_beat_band3[isum]
	  sum_calm_band1 += corr_calm_band1[isum]
	  sum_calm_band2 += corr_calm_band2[isum]
	  sum_calm_band3 += corr_calm_band3[isum]
	  
    sum1 = sum_beat_band1/24
    sum2 = sum_beat_band2/24
    sum3 = sum_beat_band3/24
    
    sum4 = sum_calm_band1/24
    sum5 = sum_calm_band2/24
    sum6 = sum_calm_band3/24

    
    
    ## PLOT-PART 1

    # plot the stacked beat-xcorr-traces for BAND1 and the average (out of sum1)
    plt.figure()
    ax = plt.subplot(111)
    for ipl in range(0,24):
	  ax.plot(time_vector,corr_beat_band1[ipl]/np.max(np.abs(corr_beat_band1[ipl])) + 1 + ipl)
    ax.plot(time_vector,sum1/np.max(np.abs(sum1)),linewidth = 3)
    plt.savefig(outDir +"beat_0-2Hz" + str(ch1) + '_' + str(ch2))

    # plot the stacked beat-xcorr-traces for BAND2 and the average (out of sum1)	  
    plt.figure()
    ax = plt.subplot(111)
    for ipl in range(0,24):
	  ax.plot(time_vector,corr_beat_band2[ipl]/np.max(np.abs(corr_beat_band2[ipl])) + 1 + ipl)
    ax.plot(time_vector,sum2/np.max(np.abs(sum2)),linewidth = 3)
    plt.savefig(outDir +"beat_2-8Hz" + str(ch1) + '_' + str(ch2))
    
    # plot the stacked beat-xcorr-traces for BAND3 and the average (out of sum1)	  
    plt.figure()
    ax = plt.subplot(111)
    for ipl in range(0,24):
	  ax.plot(time_vector,corr_beat_band3[ipl]/np.max(np.abs(corr_beat_band3[ipl])) + 1 + ipl)
    ax.plot(time_vector,sum3/np.max(np.abs(sum3)),linewidth = 3)
    plt.savefig(outDir +"beat_8-24Hz" + str(ch1) + '_' + str(ch2))
    
    
    
    ## PLOT-PART 2	  
    
    # plot the stacked calm-xcorr-traces for BAND1 and the average (out of sum2)
    plt.figure()
    ax = plt.subplot(111)
    for ipl2 in range(0,24):
	  ax.plot(time_vector,corr_calm_band1[ipl2]/np.max(np.abs(corr_calm_band1[ipl2])) + 1 + ipl2)
    ax.plot(time_vector,sum4/np.max(np.abs(sum4)),linewidth = 3)
    plt.savefig(outDir +"calm_0-2Hz" + str(ch1) + '_' + str(ch2))
    
    # plot the stacked calm-xcorr-traces for BAND2 and the average (out of sum2)
    plt.figure()
    ax = plt.subplot(111)
    for ipl2 in range(0,24):
	  ax.plot(time_vector,corr_calm_band2[ipl2]/np.max(np.abs(corr_calm_band2[ipl2])) + 1 + ipl2)
    ax.plot(time_vector,sum5/np.max(np.abs(sum5)),linewidth = 3)
    plt.savefig(outDir +"calm_2-8Hz" + str(ch1) + '_' + str(ch2))
    
    # plot the stacked calm-xcorr-traces for BAND3 and the average (out of sum2)
    plt.figure()
    ax = plt.subplot(111)
    for ipl2 in range(0,24):
	  ax.plot(time_vector,corr_calm_band3[ipl2]/np.max(np.abs(corr_calm_band3[ipl2])) + 1 + ipl2)
    ax.plot(time_vector,sum6/np.max(np.abs(sum6)),linewidth = 3)
    plt.savefig(outDir +"calm_8-24Hz" + str(ch1) + '_' + str(ch2))
	  
	  
	  

	  
	  