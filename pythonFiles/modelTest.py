from netCDF4 import Dataset
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

fldLoc = '.' #'/scratch/srai6/MOM6/postProcessing/modelTruthComparison'
fileName = 'prog_100_instants.nc'
ds = Dataset(fldLoc + '/' + fileName)


def get_Kernel(filterLength, gridSizeX, gridSizeY):
    #All the inputs are to be given in KM

    kernelSizeX = filterLength // gridSizeX + 1

    kernelSizeY = filterLength // gridSizeY + 1

    print('kernel size for Filtering ', kernelSizeX, kernelSizeY)
    print('Filtering to ', filterLength,
          ' km, gridSizeIn Km =', gridSizeX, gridSizeY)

    xx = np.arange(0, (kernelSizeX + 3)*gridSizeX, float(gridSizeX))
    yy = np.arange(0, (kernelSizeY + 3)*gridSizeY, float(gridSizeY))

    xx = xx - xx.mean()
    yy = yy - yy.mean()
    XX, YY = np.meshgrid(xx, yy)
    RR = np.sqrt(XX**2 + YY**2)

    weight = 0.5-0.5*np.tanh((abs(RR)-filterLength/2)/10.0)
    kernel = weight/np.sum(weight)

    del xx, yy, XX, YY, RR, weight

    return kernel


def get_filtered_Field(Pa, filterLength, gridSizeX, gridSizeY):

    (timeLen, Ylen, Xlen) = np.shape(Pa)

    PaBar = np.zeros((timeLen, Ylen, Xlen), dtype=float)

    kernel = get_Kernel(filterLength, gridSizeX, gridSizeY)
    for i in range(timeLen):
        PaBar[i, :, :] = signal.convolve2d(
            Pa[i, :, :], kernel, mode='same', boundary='wrap')

    return PaBar


def getInterpolated(field, oldX, oldY, newX, newY):
    timelen = np.shape(field)[0]
    Xlen = len(newX)
    Ylen = len(newY)
    fieldnew = np.zeros((timelen, Ylen, Xlen), dtype=float)
    for i in range(timelen):
        f = interpolate.interp2d(oldX, oldY, field[i, :, :], kind='linear')
        fieldnew[i, :, :] = f(newX, newY)

    return fieldnew


def getGradient(field, dx, dy):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    gradx = (np.roll(field, -1, axis=xaxis) -
             np.roll(field, 1, axis=xaxis))/(2*dx)
    grady = (np.roll(field, -1, axis=yaxis) -
             np.roll(field, 1, axis=yaxis))/(2*dy)

    return gradx, grady


xqNC = ds.variables['xq']
yqNC = ds.variables['yq']
xhNC = ds.variables['xh']
yhNC = ds.variables['yh']

u = np.array(ds.variables['u'])[:, 0, :, :]  # (xq, yh)
v = np.array(ds.variables['v'])[:, 0, :, :]  # (xh, yq)
h = np.array(ds.variables['h'])[:, 0, :, :]  # (xh, yh)

xq = np.array(xqNC)
yq = np.array(yqNC)

xh = np.array(xhNC)
yh = np.array(yhNC)

dxH = xh[1] - xh[0]
dyH = yh[1] - yh[0]

dxQ = xq[1] - xq[0]
dyQ = yq[1] - yq[0]

print(dxH, dyH, dxQ, dyQ)

U = getInterpolated(u, xq, yh, xh, yh)
V = getInterpolated(v, xh, yq, xh, yh)

hU = h * U
hV = h * V

ellList = np.arange(50, 850, 50)
ellLen = len(ellList)
Ylen, Xlen = len(yh), len(xh)

timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

timeLen = len(timeVal)

allUbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
allVbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
allhbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
allhUbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
allhVbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)

for ellIndx in range(ellLen):
    ell = ellList[ellIndx]
    allUbar[ellIndx, :, :, :] = get_filtered_Field(U, ell, dxH, dyH)
    allVbar[ellIndx, :, :, :] = get_filtered_Field(V, ell, dxH, dyH)
    allhbar[ellIndx, :, :, :] = get_filtered_Field(h, ell, dxH, dyH)
    allhUbar[ellIndx, :, :, :] = get_filtered_Field(hU, ell, dxH, dyH)
    allhVbar[ellIndx, :, :, :] = get_filtered_Field(hV, ell, dxH, dyH)


writeFileName = 'filteredFieldsForPhilipsTauAndModel.nc'
writeDS = Dataset(writeFileName, 'w', format='NETCDF4_CLASSIC')

writeDS.createDimension('Time', None)
writeDS.createDimension('xh', Xlen)
writeDS.createDimension('yh', Ylen)
writeDS.createDimension('ell', ellLen)

wcdf_Xh = writeDS.createVariable('xh', np.float32, ('xh'))
wcdf_Xh.long_name = 'h point nominal longitude'
wcdf_Xh.units = 'kilometers'
wcdf_Xh[:] = xh[:]

wcdf_Yh = writeDS.createVariable('yh', np.float32, ('yh'))
wcdf_Yh.long_name = 'h point nominal latitude'
wcdf_Yh.units = 'kilometers'
wcdf_Yh[:] = yh[:]

wcdf_Time = writeDS.createVariable('Time', np.float32, ('Time'))
wcdf_Time.long_name = "Time"
wcdf_Time.units = timeUnits
wcdf_Time.cartesian_axis = "T"
wcdf_Time.calendar_type = "JULIAN"
wcdf_Time.calendar = "JULIAN"
wcdf_Time[:] = timeVal

wcdf_ell = writeDS.createVariable('ell', np.int16, ('ell'))
wcdf_ell.long_name = 'filterLength'
wcdf_ell.units = 'kilometers'
wcdf_ell[:] = ellList


wcdf_Ubar = writeDS.createVariable(
    'U', np.float32, ('ell', 'Time', 'yh', 'xh'))
wcdf_Ubar.long_name = "Zonal Velocity"
wcdf_Ubar.units = "ms-1"
wcdf_Ubar[:, :, :, :] = allUbar[:, :, :, :]

wcdf_Vbar = writeDS.createVariable(
    'V', np.float32, ('ell', 'Time', 'yh', 'xh'))
wcdf_Vbar.long_name = "Meridional Velocity"
wcdf_Vbar.units = "ms-1"
wcdf_Vbar[:, :, :, :] = allVbar[:, :, :, :]

wcdf_hbar = writeDS.createVariable(
    'h', np.float32, ('ell', 'Time', 'yh', 'xh'))
wcdf_hbar.long_name = "Top Layer Height"
wcdf_hbar.units = "m"
wcdf_hbar[:, :, :, :] = allhbar[:, :, :, :]

wcdf_hUbar = writeDS.createVariable(
    'hU_bar', np.float32, ('ell', 'Time', 'yh', 'xh'))
wcdf_hUbar.long_name = "product of h and U filtered"
wcdf_hUbar.units = "m^2/sec"
wcdf_hUbar[:, :, :, :] = allhUbar[:, :, :, :]

wcdf_hVbar = writeDS.createVariable(
    'hV_bar', np.float32, ('ell', 'Time', 'yh', 'xh'))
wcdf_hVbar.long_name = "product of h and V filtered"
wcdf_hVbar.units = "m^2/sec"
wcdf_hVbar[:, :, :, :] = allhVbar[:, :, :, :]

writeDS.close()
