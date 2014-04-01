#! /usr/bin/env python



def merge_single(nch,dstart,dend):
  '''Merges traces of one channel to larger traces. Used for cross-correlation'''

  # here you load all the functions you need to use
  from obspy.seg2.seg2 import readSEG2
  from obspy.core import Stream


  dataDir2 = "/import/neptun-radler/STEINACH_feb/"
  dataDir = "/import/three-data/hadzii/STEINACH/STEINACH_longtime/"
  outdir = "/home/jsalvermoser/Desktop/Processing/out_merged"

  tr = []

  for k in range(dstart, dend, 1):
    fname = '%d' %(k)
    fileName = fname + ".dat" 
    st = readSEG2(dataDir + fileName)
    #st.detrend('linear')
    tr.append(st[nch-1])


  
  new_stream = Stream(traces=tr)
  new_stream.merge(method=1, fill_value='interpolate')

  start = new_stream[0].stats.starttime
  end = new_stream[0].stats.endtime

  timeframe = str(nch)+ "_" + str(start.year) +'.'+ str(start.julday) +'.'+ str(start.hour) +'.'+ str(start.minute) +'.'+ str(start.second) \
      +'-'+ str(end.year) +'.'+ str(end.julday) +'.'+ str(end.hour) +'.'+ str(end.minute) +'.'+ str(end.second)

  new_stream.write(outdir + timeframe + ".mseed", format="MSEED")

  return new_stream[0]



