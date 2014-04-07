#! /usr/bin/env python



def merge(nch,dstart,dend):
  
  ''' read the file 'fileName' that is in directory 'dataDir' into 
      a stream called 'st'

      The i-loop is over the channels/geophones, so the merged data of i geophones
      is plotted in the end.
      The k to m loops are over the data files, so traces from files 1 to 113 are 
      merged together which equals one hour of measurements.

      For each channel i the corresponding trace is picked from all the 113 files
      and then written into a new stream 'new_stream'. In this stream all 
      traces (consecutive in time) are merged then into one long trace using the 
      'merge-method 1'. As the new_stream only consists of one trace the index '[0]'
      can always be used.

      't' is an array, built to merge a combined time axis for all the merged traces.
      The new_stream data is finally plotted against the time axis. This procedure
      is iterated for each channel, whereas subplots are built. '''

  # here you load all the functions you need to use
  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream
  import matplotlib.pyplot as plt
  import numpy as np
  from numpy import matrix
  import time

  time_start = time.clock()

  dataDir = "/home/johannes/~home/Steinach/DEC_JAN/DEC_night/"
  outdir = "/home/johannes/~home/Steinach/outmseed/"

  ax = plt.subplot(111)

  TR = []

  for a in range(0,nch):
    TR.append([])

  for k in range(dstart, dend, 1):
    fname = '%d' %(k)
    fileName = fname + ".dat" 
    st = readSEG2(dataDir + fileName)
    for i in range(0,nch):
      TR[i].append(st[i])
      
      
  for m in range(0,nch):
    new_stream = Stream(traces=TR[m])
    new_stream.merge(method=0, fill_value='interpolate')

    start = new_stream[0].stats.starttime
    end = new_stream[0].stats.endtime

    timeframe = str(m+1)+ "_" + str(start.year) +'.'+ str(start.julday) +'.'+ str(start.hour) +'.'+ str(start.minute) +'.'+ str(start.second) \
      +'-'+ str(end.year) +'.'+ str(end.julday) +'.'+ str(end.hour) +'.'+ str(end.minute) +'.'+ str(end.second)

    new_stream.write(outdir + "CH" + str(m+1) + "/" + timeframe + ".mseed", format="MSEED")

    new_stream[0].normalize()
    dt = new_stream[0].stats.starttime.timestamp


    
    t = np.linspace(new_stream[0].stats.starttime.timestamp - dt,
     new_stream[0].stats.endtime.timestamp -dt, new_stream[0].stats.npts)

    ax.plot(t, new_stream[0].data + 1.5* m)

  time_elapsed = (time.clock() - time_start)
  print time_elapsed
  plt.show()


