#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from obspy.signal.cross_correlation import xcorr
import os
import glob
import cPickle as pickle
from obspy.core import UTCDateTime
from obspy.signal.filter import bandpass
import datetime
from obspy.core import read, UTCDateTime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, DateFormatter


# STRETCHING CODE

def stretching(signalRef, signalStr, epsilons, timevec, starttime=None, endtime=None):
	"""
	Calculates the stretching factor eps. This is the factor with which a signal (signalStr)
	must be stretched to get the highest correlation with a reference signal (signalRef).
	The factor eps is chosen from an array epsilons. The time vector of both signals is timevec.
	If starttime and endtime for a time window are provided, eps is calcutlated for the positive
	and negative time window as well for both time windows together. Starttime and endtime refer
	to the positive time window; from that a negative time window is calculated
	(e.g. starttime = 20.0, endtime = 50.0 --> -20.0 and -50.0 for the negative window).
	If no starttime and endtime are given, eps is computed for the whole data.
	"""

   
	if starttime!=None and endtime!=None: # eps for time windows

		if endtime > timevec[-1]:
			raise ValueError('Time window exceeds bound of time vector!')
		if starttime < 0.0:
			raise ValueError('Positive and negative time window are overlapping!')
		if starttime > endtime:
			raise ValueError('Starttime must be smaller than endtime!')	

		# indices of starttime and endtime of the time windows

		pos_t1 = np.abs(timevec-starttime).argmin()
		pos_t2 = np.abs(timevec-endtime).argmin()

		
		# taper the time windows
		pos_time = timevec[pos_t1:(pos_t2+1)]
		pos_taper_percentage = 0.1
		pos_taper = np.blackman(int(len(pos_time) * pos_taper_percentage))
		pos_taper_left, pos_taper_right = np.array_split(pos_taper, 2)
		pos_taper = np.concatenate([pos_taper_left, np.ones(len(pos_time)-len(pos_taper)), pos_taper_right])


		pos_signalRef = pos_taper * signalRef[pos_t1:(pos_t2+1)]
		pos_signalStr = pos_taper * signalStr[pos_t1:(pos_t2+1)]

		
		# calculate the correlation coefficient CC for each epsilon
		posCC = []

		for i in xrange(0,len(epsilons),1):
			# positive time window
			pos_time_new = (1.0-epsilons[i])*pos_time
			pos_s = InterpolatedUnivariateSpline(pos_time_new, pos_signalStr)
			pos_stretch = pos_s(pos_time)
			pos_coeffs = xcorr(pos_stretch,pos_signalRef,0)
			posCC.append(abs(pos_coeffs[1]))

			
		# determine the max. CC and corresponding epsilon
		posmaxCC = max(posCC)
		posindex = posCC.index(posmaxCC)
		poseps = epsilons[posindex]


		
	
		# decomment for showing plot of signal, reference signal and stretched signal in positive timewindow
		# pos_time_eps = (1.0-poseps)*pos_time
		# s_poseps = InterpolatedUnivariateSpline(pos_time_eps, pos_signalStr)
		# stretch_poseps = s_poseps(pos_time)

		# plt.plot(pos_time, pos_signalStr, 'r')
		# plt.plot(pos_time, pos_signalRef,'b')
		# plt.plot(pos_time, stretch_poseps,'g')
		# plt.xlabel('seconds')
		# plt.title('Comparison of CCFs')
		# plt.legend(('signal', 'reference signal', 'stretched signal'))
		# plt.show()



		# decomment for showing plot of signal, reference signal and stretched signal in both timewindows
		# both_time_eps = (1.0-botheps)*both_time
		# s_botheps = InterpolatedUnivariateSpline(both_time_eps, both_signalStr)
		# stretch_botheps = s_botheps(both_time)

		# plt.plot(both_time, both_signalStr, 'r')
		# plt.plot(both_time, both_signalRef,'b')
		# plt.plot(both_time, stretch_botheps,'g')
		# plt.xlabel('seconds')
		# plt.title('Comparison of CCFs')
		# plt.legend(('signal', 'reference signal', 'stretched signal'))
		# plt.show()

		return poseps, posmaxCC
	
	elif (starttime == None and endtime != None) or (starttime != None and endtime == None):
		raise SyntaxError('Both starttime and endtime must be given!')


	else: # eps for whole data
		counter=0
		# taper the signal and the reference
		taper_percentage = 0.1
		taper = np.blackman(int(len(timevec) * taper_percentage))
		taper_left, taper_right = np.array_split(taper, 2)
		taper = np.concatenate([taper_left, np.ones(len(timevec)-len(taper)), taper_right])

		signalStr = signalStr * taper
		signalRef = signalRef * taper
		
		# calculate the correlation coefficient CC for each epsilon
		CC = []
		for i in xrange(0,len(epsilons),1):
			time_new = (1.0-epsilons[i])*timevec
			s = InterpolatedUnivariateSpline(time_new, signalStr)
			stretch = s(timevec)
			coeffs = xcorr(stretch,signalRef,0)
			CC.append(abs(coeffs[1]))
			counter+=1
			#print (counter)
		
		# determine the max. CC and corresponding epsilon
		maxCC = max(CC)
		index = CC.index(maxCC)
		eps = epsilons[index]
		
		## decomment for showing plot of signal, reference signal and stretched signal
		#time_eps = (1.0-eps)*timevec
		#s_eps = InterpolatedUnivariateSpline(time_eps, signalStr)
		#stretch_eps = s_eps(timevec)

		##plot of the signal, reference signal and the stretched signal
		#plt.plot(timevec, signalStr, 'r')
		#plt.plot(timevec, signalRef,'b')
		#plt.plot(timevec, stretch_eps,'g')
		#plt.xlabel('seconds')
		#plt.title('Comparison of CCFs')
		#plt.legend(('signal', 'reference signal', 'stretched signal'))
		#plt.show()
	
		return eps, maxCC



################################################################################################################
################################################################################################################



# PERFORM STRETCHING

import numpy as np
import matplotlib.pyplot as plt
import glob


ch1=6
ch2=47
day=8
sum1=0
month="JAN"
band = "2-8Hz"
mode1 = "beat"
mode2 = "calm"



#DIRECTORIES
#dataDir = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/"
ax = plt.subplot(111)

for ch2 in range(7,10,1):
  path1 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode1 + "_" + band +  "_CH" + str(ch2) + ".npy"
  path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode2 + "_" + band +  "_CH" + str(ch2) + ".npy"
  epsilons=np.linspace(-0.003, 0.003, 101)
  time_vector = np.linspace(-50.0,50.0,50001)
  epsis=[]
  coefs=[]
  for fname in glob.glob(path2):
    f=np.load(fname)
    if f[0]==0:
      sum1+=0
    else:
      sum1 += f/np.max(np.abs(f))
  ref= sum1/np.max(np.abs(sum1)) 

  for fname in glob.glob(path1):
    corr2 = np.load(fname)
    start=0.05   # start after the last surface wave went through
    end=11.5      # (11.5s) end when 75% of the signal already decayed
    if corr2[0]!=0:
      eps, maxCC = stretching(ref, corr2, epsilons, time_vector,starttime=start, endtime=end)
      print eps, maxCC
      epsis.append(eps)
      coefs.append(maxCC)
    else:
      epsis.append(0)
      coefs.append(0)
    
  ax.plot(epsis, color='b')
  
print 'part1 completed'  
  
for ch2 in range(16,20,2):
  path1 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode1 + "_" + band +  "_CH" + str(ch2) + ".npy"
  path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode2 + "_" + band +  "_CH" + str(ch2) + ".npy"
  epsilons=np.linspace(-0.003, 0.003, 101)
  time_vector = np.linspace(-50.0,50.0,50001)
  epsis=[]
  coefs=[]
  for fname in glob.glob(path2):
    f=np.load(fname)
    if f[0]==0:
      sum1+=0
    else:
      sum1 += f/np.max(np.abs(f))
  ref= sum1/np.max(np.abs(sum1)) 

  for fname in glob.glob(path1):
    corr2 = np.load(fname)
    if corr2[0]!=0:
      start=0.05    # start after the last surface wave went through
      end=15.0      # (15s) end when 75% of the signal already decayed
      eps, maxCC = stretching(ref, corr2, epsilons, time_vector,starttime=start, endtime=end)
      print eps, maxCC
      epsis.append(eps)
      coefs.append(maxCC)
    else:
      epsis.append(0)
      coefs.append(0)
    
  ax.plot(epsis, color='g')
  
print 'part2 completed'   
  
  
for ch2 in range(45,48,1):
  path1 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode1 + "_" + band +  "_CH" + str(ch2) + ".npy"
  path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode2 + "_" + band +  "_CH" + str(ch2) + ".npy"
  epsilons=np.linspace(-0.003, 0.003, 101)
  time_vector = np.linspace(-50.0,50.0,50001)
  epsis=[]
  coefs=[]
  for fname in glob.glob(path2):
    f=np.load(fname)
    if f[0]==0:
      sum1+=0
    else:
      sum1 += f/np.max(np.abs(f))
  ref= sum1/np.max(np.abs(sum1)) 

  for fname in glob.glob(path1):
    corr2 = np.load(fname)
    if corr2[0]!=0:
      start=0.05    # start after the last surface wave went through
      end=20.0      # (20s) end when 75% of the signal already decayed
      eps, maxCC = stretching(ref, corr2, epsilons, time_vector ,starttime=start, endtime=end)
      print eps, maxCC
      epsis.append(eps)
      coefs.append(maxCC)
    else:
      epsis.append(0)
      coefs.append(0)
    
  ax.plot(epsis, color='r')
  
print 'part3 completed'   

  
for ch2 in range(27,33,2):
  path1 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode1 + "_" + band +  "_CH" + str(ch2) + ".npy"
  path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode2 + "_" + band +  "_CH" + str(ch2) + ".npy"
  epsilons=np.linspace(-0.003, 0.003, 101)
  time_vector = np.linspace(-50.0,50.0,50001)
  epsis=[]
  coefs=[]
  for fname in glob.glob(path2):
    f=np.load(fname)
    if f[0]==0:
      sum1+=0
    else:
      sum1 += f/np.max(np.abs(f))
  ref= sum1/np.max(np.abs(sum1)) 

  for fname in glob.glob(path1):
    corr2 = np.load(fname)
    if corr2[0]!=0:
      start=0.05    # start after the last surface wave went through
      end=18.0      # (18s) end when 75% of the signal already decayed
      eps, maxCC = stretching(ref, corr2, epsilons, time_vector,starttime=start, endtime=end)
      print eps, maxCC
      epsis.append(eps)
      coefs.append(maxCC)
    else:
      epsis.append(0)
      coefs.append(0)
    
  ax.plot(epsis, color='k')
  
print 'part4 completed'  

  
#for ch2 in range(35,41,2):
  #path1 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode1 + "_" + band +  "_CH" + str(ch2) + ".npy"
  #path2 = "/home/jsalvermoser/Desktop/Processing/bands_SNR/" + "CH" + str(ch1) + "_CH" + str(ch2) + "/" + month + str(day) + "/*" + mode2 + "_" + band +  "_CH" + str(ch2) + ".npy"
  #epsilons=np.linspace(-0.003, 0.003, 101)
  #time_vector = np.linspace(-50.0,50.0,50001)
  #epsis=[]
  #coefs=[]
  #for fname in glob.glob(path2):
    #f=np.load(fname)
    #if f[0]==0:
      #sum1+=0
    #else:
      #sum1 += f/np.max(np.abs(f))
  #ref= sum1/np.max(np.abs(sum1)) 

  #for fname in glob.glob(path1):
    #corr2 = np.load(fname)
    #if corr2[0]!=0:
      #start=0.05    # start after the last surface wave went through
      #end=19.0      # (18s) end when 75% of the signal already decayed
      #eps, maxCC = stretching(ref, corr2, epsilons, time_vector,starttime=start, endtime=end)
      #print eps, maxCC
      #epsis.append(eps)
      #coefs.append(maxCC)
    #else:
      #epsis.append(0)
      #coefs.append(0)
    
  #ax.plot(epsis, color='purple')
  
  
plt.show()
  



