"""import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import numpy

def makeData ():
    x = numpy.arange (-10, 10, 0.1)
    y = numpy.arange (-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)

    zgrid = numpy.sin (xgrid) * numpy.sin (ygrid) / (xgrid * ygrid)
    return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

axes.plot_surface(x, y, z, rstride=3, cstride=3, cmap = LinearSegmentedColormap.from_list ("red_blue", ['b', 'w', 'r'], 256))
print(x.size,x.shape,y.size,z.size,z.shape)
pylab.show()
"""
import numpy as np
from numpy.linalg import eig
import cmath
import math
import sys
from functools import reduce
import pandas as pd
import pylab as pl
import array
from scipy.fftpack import fft, ifft

dataDC2 = np.loadtxt("testupto30up.txt", comments='%')#, usecols=(0,1,3),unpack=True)
fieldDC2 = np.zeros(29)
freqDC2 = (dataDC2[650:1415,0])/1e6
freqStartDC2 = freqDC2[0]
NumPoints = 765
freqStopDC2 = freqDC2[764]
freqStepDC2 = freqDC2[11]-freqDC2[10]
IntensityDC2 = np.zeros((29,765))
for i in range(29):
	fieldDC2[i] = np.mean(dataDC2[i*5000:(i+1)*5000,1])
	IntensityDC2[i,:] = dataDC2[i*5000+650:i*5000+1415,3]

pl.figure()
pl.pcolor(freqDC2, fieldDC2, IntensityDC2)
pl.xlabel(" Frequency")
pl.ylabel(" B (T)")
pl.show()