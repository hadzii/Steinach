# -*- coding: utf-8 -*-


# plots SNR for cross-correlation functions with steadily growing window lengths
# starting at 32 seconds (one file) and ending in somewhat around 73 files merged (40 mins)

import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
from obspy import UTCDateTime
from matplotlib.dates import DayLocator, HourLocator,DateFormatter, MONDAY, WeekdayLocator
import csv,codecs,cStringIO


temperature = np.genfromtxt('temperatures_FEB_MAR.csv', delimiter=';')
temp=[]
dates=[]
data=[]

f=open('dates_FEB_MAR.csv','rb')
reader=csv.reader(f)
for row in reader:
	data.append(row)

for i in range(0,984):
	dates.append(UTCDateTime(data[i][0]).datetime)

for k in range(1,985):
  temp.append(temperature[k][3])



print dates[0]
#dates=[UTCDatetime(x[0].datetime for x in dates]

days=DayLocator(interval=2)
print days
hours=HourLocator()
weekFormatter = DateFormatter('%b %d')
#mondays = WeekdayLocator(MONDAY)
#dayFormatter = DateFormatter('%d')

plt.plot(dates,temp)
ax=plt.gca()
ax.xaxis.set_major_locator(days)
#ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(hours)
ax.xaxis.set_major_formatter(weekFormatter)
plt.setp( plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.xlabel('date', fontsize=12, fontweight='bold')
plt.ylabel(u'temperature [°C]', fontsize=12, fontweight='bold')
plt.title('Temperature data for February and March \n', fontsize=18, fontweight='bold')
plt.show()

