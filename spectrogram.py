#! /usr/bin/env python

from obspy import read
from merge_single import merge_single

tr = merge_single(6,1000001,1000675)

# lowpass-filter the crossc_beat correlation function 

tr.filter('lowpass', freq = 24, zerophase=True, corners=8)
tr.filter('bandstop', freqmin=8, freqmax=14, corners=2, zerophase=True)

tr.spectrogram(wlen=2, per_lap=0.5, mult=1, dbscale=True)