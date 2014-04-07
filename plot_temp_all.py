# -*- coding: utf-8 -*-


# plots SNR for cross-correlation functions with steadily growing window lengths
# starting at 32 seconds (one file) and ending in somewhat around 73 files merged (40 mins)

import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
from obspy import UTCDateTime
from matplotlib.dates import DayLocator, HourLocator,DateFormatter

temperature = np.genfromtxt('temperatures_FEB_MAR.csv', delimiter=';')
temp=[]
dates=[]

for k in range(1,985):
  temp.append(temperature[k][3])
#  dates.append(UTCDateTime(temperature[k][2]))
#  print UTCDateTime(temperature[k][2])

#date=[x.datetime for x in dates]
  
#days=DayLocator()
#hours=HourLocator()

plt.plot(temp)
#ax=plt.gca()
#ax.xaxis.set_major_locator(days)
#ax.xaxis.set_minor_locator(hours)
#ax.xaxis.set_major_formatter(DateFormatter('%d'))

plt.show()

#time = np.linspace(1,73,73)

#time=time*32
#x=np.arange(0,73*32,100)
#print x
#plt.plot(time,SNR, color='g', linewidth=4)
#plt.title('Signal-to-Noise ratio of merged files (given time_windows)', fontweight='bold', fontsize=20)
#plt.ylabel('signal-to-noise ratio', fontsize=16)
#plt.xlabel('time-window [s]', fontsize=16)
#plt.xlim(0,75*32)
#plt.xticks(x)
#plt.show() 