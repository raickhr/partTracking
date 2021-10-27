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


def getZCurl(fieldx, fieldy, dx, dy):
    (timeLen, Ylen, Xlen) = np.shape(fieldx)
    yaxis = 1
    xaxis = 2

    gradx = (np.roll(fieldy, -1, axis=xaxis) -
             np.roll(fieldy, 1, axis=xaxis))/(2*dx)
    grady = (np.roll(fieldx, -1, axis=yaxis) -
             np.roll(fieldx, 1, axis=yaxis))/(2*dy)

    return (gradx - grady)


def getLaplacian(field, dx, dy):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    d2dx2 = (np.roll(field, -1, axis=xaxis) - 2*field +
             np.roll(field, 1, axis=xaxis))/(dx**2)
    d2dy2 = (np.roll(field, -1, axis=yaxis) - 2*field +
             np.roll(field, 1, axis=yaxis))/(dy**2)

    return d2dx2 + d2dy2


def get_d4_dx4(field, dx):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    field_ip3 = np.roll(field, -3, axis=xaxis)
    field_ip2 = np.roll(field, -2, axis=xaxis)
    field_ip1 = np.roll(field, -1, axis=xaxis)
    field_im1 = np.roll(field, 1, axis=xaxis)
    field_im2 = np.roll(field, 2, axis=xaxis)
    field_im3 = np.roll(field, 3, axis=xaxis)

    returnVal =  1/(6*dx**4) * (\
                 -field_ip3  \
                 +12 * field_ip2  \
                 -39 * field_ip1  \
                 +56 * field      \
                 -39 * field_im1  \
                 +12 * field_im2  \
                 -field_im3 )

    return returnVal


def get_d4_dy4(field, dy):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    field_jp3 = np.roll(field, -3, axis=yaxis)
    field_jp2 = np.roll(field, -2, axis=yaxis)
    field_jp1 = np.roll(field, -1, axis=yaxis)
    field_jm1 = np.roll(field, 1, axis=yaxis)
    field_jm2 = np.roll(field, 2, axis=yaxis)
    field_jm3 = np.roll(field, 3, axis=yaxis)

    returnVal = 1/(6*dy**4) * (
        -field_jp3
        + 12 * field_jp2
        - 39 * field_jp1
        + 56 * field
        - 39 * field_jm1
        + 12 * field_jm2
        - field_jm3)

    return returnVal


def get_d2_dx2(field, dx):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    field_ip2 = np.roll(field, -2, axis=xaxis)
    field_ip1 = np.roll(field, -1, axis=xaxis)
    field_im1 = np.roll(field, 1, axis=xaxis)
    field_im2 = np.roll(field, 2, axis=xaxis)
    
    returnVal = 1/(12*dx**2) * (
        -1 * field_ip2
        +16 * field_ip1
        -30 * field
        +16 * field_im1
        -1 * field_im2)

    return returnVal


def get_d2_dy2(field, dy):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    field_jp2 = np.roll(field, -2, axis=yaxis)
    field_jp1 = np.roll(field, -1, axis=yaxis)
    field_jm1 = np.roll(field, 1, axis=yaxis)
    field_jm2 = np.roll(field, 2, axis=yaxis)

    returnVal = 1/(12*dy**2) * (
        -1 * field_jp2
        + 16 * field_jp1
        - 30 * field
        + 16 * field_jm1
        - 1 * field_jm2)

    return returnVal

def get_d4_dx2dy2(field, dx, dy):
    (timeLen, Ylen, Xlen) = np.shape(field)
    yaxis = 1
    xaxis = 2

    d2_dy2_field = get_d2_dy2(field,dy)
    returnVal = get_d2_dx2(d2_dy2_field, dx)

    return returnVal


def getBiharmonic(field, dx, dy):
    returnValue = get_d4_dx4(field, dx) + get_d4_dy4(field,dy) + 2 * get_d4_dx2dy2(field,dx,dy)
    return returnValue




