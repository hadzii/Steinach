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
from obspy.seg2.seg2 import readSEG2
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

ch1=35
ch2=22
DEC_Dir= '/import/three-data/hadzii/STEINACH/Steinach/Steinach_ACTIVE/'
FEB_Dir= '/import/three-data/hadzii/STEINACH/Steinach/STEINACH_act_feb/'
DEC_fn='1002.dat'
FEB_fn='1000002.dat'

pos1D=6
pos2D=9

pos1F=6
pos2F=9

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

print st_DEC[0].stats.starttime
print st_FEB[0].stats.starttime

epsis_DEC=[]
epsis_FEB=[]

for ch in range(6,25):
  #### 1-bit normalization:
  tr1_DEC = np.sign(st_DEC[ch].data)
  tr2_DEC = np.sign(st_DEC[ch+2].data)
  tr1_FEB = np.sign(st_FEB[ch].data)
  tr2_FEB = np.sign(st_FEB[ch+2].data)


  index1, value1, corr_DEC = xcorr(tr1_DEC, tr2_DEC, 1500, full_xcorr=True)
  index2, value2, corr_FEB = xcorr(tr1_FEB, tr2_FEB, 1500, full_xcorr=True)

  #### NORMALIZE:
  corr_DEC = corr_DEC/np.max(np.abs(corr_DEC))
  corr_FEB = corr_FEB/np.max(np.abs(corr_FEB))
  ref = (corr_DEC + corr_FEB)/np.max(np.abs(corr_DEC+corr_FEB))


  #plt.plot(corr_DEC, color='r')
  #plt.plot(corr_FEB, color='b')
  #plt.plot(ref, color='k')
  #plt.show()

  #### STRETCHING:
  epsilons = np.linspace(-0.1, 0.1, 501)
  time_vector = np.linspace(-5.0,5.0,3001)

  eps_DEC, maxCC_DEC = stretching(ref, corr_DEC, epsilons, time_vector)
  eps_FEB, maxCC_FEB = stretching(ref, corr_FEB, epsilons, time_vector)
  epsis_DEC.append(eps_DEC)
  epsis_FEB.append(eps_FEB)

plt.plot(epsis_DEC,color='r')
plt.plot(epsis_FEB,color='b')
plt.show()

  



