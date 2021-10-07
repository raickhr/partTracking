from netCDF4 import Dataset
import numpy as np
from scipy import signal, interpolate


def fillInvalidWithZero(array):
    mask = abs(array) > 1e5 + np.isnan(array)
    arr = np.ma.array(array, mask = mask, fill_value=0.0).filled()
    return arr

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
    # u, v and q  are not in the same place so this function
    # interpolates to the specified grid
    timelen = np.shape(field)[0]
    Xlen = len(newX)
    Ylen = len(newY)
    fieldnew = np.zeros((timelen, Ylen, Xlen), dtype=float)
    for i in range(timelen):
        #f = interpolate.interp2d(oldX, oldY, field[i, :, :], kind='linear')
        f = interpolate.RectBivariateSpline(
            oldY, oldX, field[i, :, :], kx = 4, ky =4)
        #fieldnew[i, :, :] = f(newX, newY)
        fieldnew[i, :, :] = f(newY, newX)

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


def getDiv(fieldx, fieldy, dx, dy):
    (timeLen, Ylen, Xlen) = np.shape(fieldx)
    yaxis = 1
    xaxis = 2

    gradx = (np.roll(fieldx, -1, axis=xaxis) -
             np.roll(fieldx, 1, axis=xaxis))/(2*dx)
    grady = (np.roll(fieldy, -1, axis=yaxis) -
             np.roll(fieldy, 1, axis=yaxis))/(2*dy)

    return (gradx + grady)

