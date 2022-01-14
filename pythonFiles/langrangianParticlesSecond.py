from os import write
import matplotlib.pyplot as plt
import numpy as np
import argparse
from netCDF4 import Dataset
import sys
from pathlib import Path

interval = 3600

def interpolate(Q1, Q2, Q3, Q4, x, y, dx, dy):
    ### this function bilinearly interpolates inside a quad
    #
    #             (4)---------(3)
    #              |           |
    #              |           |
    #              |           |
    #             (1)---------(2)
    #
    # Q1, Q2, Q3 and Q4 are the function values at nodes 1,2,3 and 4
    # x, y is a co-ordinate inside the quad where the value is to be interpolated

    dx *= 1000
    dy *= 1000

    x1, y1 = 0, 0
    x2, y2 = dx, 0
    x3, y3 = dx, dy
    x4, y4 = 0, dy

    # shape function
    N1 = 1/(dx*dy) * (x - x2) * (y - y4)
    N2 = -1/(dx * dy) * (x - x1) * (y - y3)
    N3 = 1/(dx * dy) * (x - x4) * (y - y2)
    N4 = -1/(dx * dy) * (x - x3) * (y - y1)

    return(N1 * Q1 + N2 * Q2 + N3 * Q3 + N4 * Q4)


parser = argparse.ArgumentParser()

parser.add_argument("--nParticles", "-n", type=int, default=500, action='store',
                    help="number of particles to be tracked")
parser.add_argument("--file", "-f", type=str, action='store', required=True,
                    help="file from where the particles will be tracked")
parser.add_argument("--filteredFile", "-ff", type=str, action='store', required=True,
                    help="file from where the particles will be tracked")
parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")

args = parser.parse_args()

fileName = args.file
nParticles = args.nParticles
filteredFile = args.filteredFile
fldLoc = args.fldLoc

fileNum = int(fileName.lstrip('prog_').rstrip('_RequiredFieldsOnly.nc'))
nextFileNum = fileNum + 1
nextFile = fldLoc + '/prog_{0:03d}_RequiredFieldsOnly.nc'.format(nextFileNum)
nextFilePath = Path(nextFile)

wFname = fldLoc + \
    '/prog_{0:03d}_pTracking_{1:03d}p_hrlyStride.nc'.format(fileNum, nParticles)

prevFileNum = fileNum - 1
prevFile = fldLoc + \
    '/prog_{0:03d}_pTracking_{1:03d}p_hrlyStride.nc'.format(
        prevFileNum, nParticles)
prevFilePath = Path(prevFile)

ds = Dataset(fileName)
ds2 = Dataset(filteredFile)

xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])

dx = xh[1] - xh[0]
dy = yh[1] - yh[0]

xlen = len(xh)
ylen = len(yh)

timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units
timelen = len(timeVal)
dt = (timeVal[1] - timeVal[0])  # *24*3600
stride = int(interval//dt)
dt = dt * stride
substeps = 20

print('stride', stride, 'dt', dt)

disp_x = np.array(ds.variables['disp_x'])
disp_y = np.array(ds.variables['disp_y'])

Lambda = np.array(ds2.variables['Lambda'])
Pi = np.array(ds2.variables['Pi'])

u = np.array(ds.variables['u'])
v = np.array(ds.variables['v'])

if nextFilePath.is_file():
    print('continuing file present')
    ds3 = Dataset(nextFile)
    # (s1, s2, s3) = np.shape(u)
    # u1 = np.zeros((stride, s2, s3), dtype=np.float64)
    # v1 = u1.copy()
    ## need future u1 values for the last step
    u1 = np.array(ds3.variables['u'])[0:stride, :, :]
    v1 = np.array(ds3.variables['v'])[0:stride, :, :]
    u = np.concatenate((u, u1), axis=0)
    v = np.concatenate((v, v1), axis=0)
    ## put dummy values for Lambda and Pi so stack overflow does not occur
    dummy = np.zeros(np.shape(u1), dtype=np.float64)
    Lambda = np.concatenate((Lambda, dummy), axis=0)
    Pi = np.concatenate((Pi, dummy), axis=0)


else:
    timelen -= stride

## generate random particles
xpos = np.random.rand(nParticles) * (xh[xlen-1] - xh[0]) + xh[0]
ypos = np.random.rand(nParticles) * (yh[ylen-1] - yh[0]) + yh[0]

## get first position
firstDS = Dataset(
    fldLoc + '/prog_{0:03d}_pTracking_{1:03d}p.nc'.format(1, nParticles))
firstDSTime = np.array(firstDS.variables['Time'])
for timeNumber in range(len(firstDSTime)):
    if firstDSTime[timeNumber]%dt == 0.0:
        xpos = np.array(firstDS.variables['xpos'])[timeNumber, :]
        ypos = np.array(firstDS.variables['ypos'])[timeNumber, :]
        break

if prevFilePath.is_file():
    print('getting positions from '+prevFile)
    ds4 = Dataset(prevFile)
    xpos = np.array(ds4.variables['xpos_next'])
    ypos = np.array(ds4.variables['ypos_next'])


LambdaVal = np.zeros((nParticles), dtype=np.float64)
PiVal = np.zeros((nParticles), dtype=np.float64)


def getPiandLambda(xpos, ypos, currentLambda, currentPi):
    global xh, yh, xlen, ylen, dx, dy
    lambdaVal = -1e34
    PiVal = -1e34

    if (xpos >= xh[xlen-1]):
        xpos = (xpos - xh[xlen-1]) + xh[0]
    if (xpos < xh[0]):
        xpos = xh[xlen-1] - (xh[0] - xpos)

    if (ypos >= yh[ylen-1]) or \
        (ypos <= yh[0]) or \
        (abs(xpos) > 1e5) or \
            (abs(ypos) > 1e5):
        #print('xpos', 'ypos')
        lambdaVal = -1e34
        PiVal = -1e34

    else:
        index_X = int((xpos - xh[0])//dx)
        index_Y = int((ypos - yh[0])//dy)

        x1 = index_X
        y1 = index_Y

        # if index_X < xlen-1:
        #     x1 = index_X
        # else:
        #     x1 = 0
        # if index_Y < ylen-1:
        #     y1 = index_Y
        # else:
        #     y1 = 0

        x2 = index_X+1
        y2 = index_Y

        x3 = index_X+1
        y3 = index_Y+1

        x4 = index_X
        y4 = index_Y+1

        lambda_1 = currentLambda[y1, x1]
        lambda_2 = currentLambda[y2, x2]
        lambda_3 = currentLambda[y3, x3]
        lambda_4 = currentLambda[y4, x4]

        Pi_1 = currentPi[y1, x1]
        Pi_2 = currentPi[y2, x2]
        Pi_3 = currentPi[y3, x3]
        Pi_4 = currentPi[y4, x4]

        lambdaVal = interpolate(lambda_1,
                                lambda_2,
                                lambda_3,
                                lambda_4,
                                xpos, ypos, dx, dy)

        PiVal = interpolate(Pi_1,
                            Pi_2,
                            Pi_3,
                            Pi_4,
                            xpos, ypos, dx, dy)

    return lambdaVal, PiVal


def updatePositon(xpos, ypos,
                  lambdaVal, PiVal,
                  current_x_disp, current_y_disp,
                  currentLambda, currentPi):
    global substeps
    continueLoop = np.zeros((nParticles), dtype=bool)
    for i in range(nParticles):
        if continueLoop[i]:
            continue
        cur_xpos = xpos[i]
        cur_ypos = ypos[i]
        index_X = 0
        index_Y = 0
        for j in range(substeps):
            if (cur_xpos >= xh[xlen-1]):
                cur_xpos = (cur_xpos - xh[xlen-1]) + xh[0]
            if (cur_xpos <= xh[0]):
                cur_xpos = xh[xlen-1] - (xh[0] - cur_xpos)

            if (cur_ypos >= yh[ylen-1]) or \
                (cur_ypos <= yh[0]) or \
                (abs(cur_xpos) > 1e5) or \
                    (abs(cur_ypos) > 1e5):
                cur_xpos = -1e34
                cur_ypos = -1e34
                xpos[i] = -1e34
                ypos[i] = -1e34
                print('skipping', i)
                continueLoop[i] = True
                break

            else:
                #
                #if ypos[i] != float('nan'):
                ## indices of lower left corner of each particle
                try:
                    index_X = int((cur_xpos - xh[0])//dx)
                    index_Y = int((cur_ypos - yh[0])//dy)
                except:
                    print(i, j, cur_xpos, cur_ypos)
                    sys.exit()

                ## x-displacement values at all the four corners

                part_xdisp1 = current_x_disp[index_Y, index_X] * 1/substeps
                part_xdisp2 = current_x_disp[index_Y, index_X+1] * 1/substeps
                part_xdisp3 = current_x_disp[index_Y+1, index_X+1] * 1/substeps
                part_xdisp4 = current_x_disp[index_Y+1, index_X] * 1/substeps

                part_ydisp1 = current_y_disp[index_Y, index_X] * 1/substeps
                part_ydisp2 = current_y_disp[index_Y, index_X+1] * 1/substeps
                part_ydisp3 = current_y_disp[index_Y+1, index_X+1] * 1/substeps
                part_ydisp4 = current_y_disp[index_Y+1, index_X] * 1/substeps

                part_xdisp = interpolate(part_xdisp1,
                                         part_xdisp2,
                                         part_xdisp3,
                                         part_xdisp4,
                                         cur_xpos, cur_ypos, dx, dy)

                part_ydisp = interpolate(part_ydisp1,
                                         part_ydisp2,
                                         part_ydisp3,
                                         part_ydisp4,
                                         cur_xpos, cur_ypos, dx, dy)

                cur_xpos += part_xdisp/1000
                cur_ypos += part_ydisp/1000

        xpos[i] = cur_xpos
        ypos[i] = cur_ypos

        if continueLoop[i]:
            lambdaVal[i], PiVal[i] = -1e34, -1e34
        else:
            lambdaVal[i], PiVal[i] = getPiandLambda(
                xpos[i], ypos[i], currentLambda, currentPi)

        # if continueLoop[i]:
        #     lambdaVal[i], PiVal[i] = -1e34, -1e34
        # else:
        #     lambda_1 = currentLambda[index_Y, index_X] * 1/substeps
        #     lambda_2 = currentLambda[index_Y, index_X+1] * 1/substeps
        #     lambda_3 = currentLambda[index_Y+1, index_X+1] * 1/substeps
        #     lambda_4 = currentLambda[index_Y+1, index_X] * 1/substeps

        #     Pi_1 = currentPi[index_Y, index_X] * 1/substeps
        #     Pi_2 = currentPi[index_Y, index_X+1] * 1/substeps
        #     Pi_3 = currentPi[index_Y+1, index_X+1] * 1/substeps
        #     Pi_4 = currentPi[index_Y+1, index_X] * 1/substeps

        #     lambdaVal[i] = interpolate(lambda_1,
        #                             lambda_2,
        #                             lambda_3,
        #                             lambda_4,
        #                             xpos, ypos, dx, dy)

        #     PiVal[i] = interpolate(Pi_1,
        #                         Pi_2,
        #                         Pi_3,
        #                         Pi_4,
        #                         xpos, ypos, dx, dy)

    return(xpos, ypos, lambdaVal, PiVal)


def updatePositon_linVel(xpos, ypos,
                         lambdaVal, PiVal,
                         currentLambda, currentPi,
                         u1, v1, u2, v2):

    global dt, substeps
    continueLoop = np.zeros((nParticles), dtype=bool)
    for i in range(nParticles):
        if continueLoop[i]:
            continue
        cur_xpos = xpos[i]
        cur_ypos = ypos[i]
        for j in range(substeps):
            if (cur_xpos >= xh[xlen-1]):
                cur_xpos = (cur_xpos - xh[xlen-1]) + xh[0]
            if (cur_xpos < xh[0]):
                cur_xpos = xh[xlen-1] - (xh[0] - cur_xpos)

            if (cur_ypos >= yh[ylen-1]) or \
                (cur_ypos <= yh[0]) or \
                (abs(cur_xpos) > 1e5) or \
                    (abs(cur_ypos) > 1e5):
                cur_xpos = -1e34
                cur_ypos = -1e34
                xpos[i] = -1e34
                ypos[i] = -1e34
                print('skipping', i)
                continueLoop[i] = True
                break

            else:
                #
                #if ypos[i] != float('nan'):
                ## indices of lower left corner of each particle
                index_X = 0
                index_Y = 0
                try:
                    index_X = int((cur_xpos - xh[0])//dx)
                    index_Y = int((cur_ypos - yh[0])//dy)
                except:
                    print(i, cur_xpos, cur_ypos)
                    sys.exit()

                ## x-displacement values at all the four corners

                w2 = j/(substeps-1)
                w1 = 1 - w2

                part_xdisp1 = ((w1 * u1[index_Y, index_X]) +
                               (w2 * u2[index_Y, index_X])) * dt/substeps
                part_xdisp2 = (
                    (w1 * u1[index_Y, index_X+1]) + (w2 * u2[index_Y, index_X+1])) * dt/substeps
                part_xdisp3 = (
                    (w1 * u1[index_Y+1, index_X+1]) + (w2 * u2[index_Y+1, index_X+1])) * dt/substeps
                part_xdisp4 = (
                    (w1 * u1[index_Y+1, index_X]) + (w2 * u2[index_Y+1, index_X])) * dt/substeps

                part_ydisp1 = ((w1 * v1[index_Y, index_X]) +
                               (w2 * v2[index_Y, index_X])) * dt/substeps
                part_ydisp2 = (
                    (w1 * v1[index_Y, index_X+1]) + (w2 * v2[index_Y, index_X+1])) * dt/substeps
                part_ydisp3 = (
                    (w1 * v1[index_Y+1, index_X+1]) + (w2 * v2[index_Y+1, index_X+1])) * dt/substeps
                part_ydisp4 = (
                    (w1 * v1[index_Y+1, index_X]) + (w2 * v2[index_Y+1, index_X])) * dt/substeps

                part_xdisp = interpolate(part_xdisp1,
                                         part_xdisp2,
                                         part_xdisp3,
                                         part_xdisp4,
                                         xpos[i], ypos[i], dx, dy)

                part_ydisp = interpolate(part_ydisp1,
                                         part_ydisp2,
                                         part_ydisp3,
                                         part_ydisp4,
                                         xpos[i], ypos[i], dx, dy)

                if part_xdisp == 0.0 and part_ydisp == 0.0:
                    print('Displacement = 0 for particle ', i)

                cur_xpos += part_xdisp/1000
                cur_ypos += part_ydisp/1000

        xpos[i] = cur_xpos
        ypos[i] = cur_ypos

        if continueLoop[i]:
            lambdaVal[i], PiVal[i] = -1e34, -1e34
        else:
            lambdaVal[i], PiVal[i] = getPiandLambda(
                xpos[i], ypos[i], currentLambda, currentPi)

    return(xpos, ypos, lambdaVal, PiVal)


xposVals = np.zeros((1, nParticles, ), dtype=np.float64)
yposVals = np.zeros((1, nParticles, ), dtype=np.float64)

xpos_next = np.zeros((nParticles, ), dtype=np.float64)
ypos_next = np.zeros((nParticles, ), dtype=np.float64)

piVals = np.zeros((1, nParticles, ), dtype=np.float64)
lambdaVals = np.zeros((1, nParticles, ), dtype=np.float64)

cdfTimeVals = [] #np.zeros((1,), dtype=np.float64)

for t in range(0, timelen):
    checkTime = timeVal[t]
    print('time :', t)
    if (checkTime%dt) == 0:
        print(checkTime)
        if t < stride:
            xposVals[0, :], yposVals[0, :] = xpos, ypos
            cdfTimeVals.append(checkTime)
            #print('dataLen = ', len(cdfTimeVals))
            for p in range(nParticles):
                lambdaVals[0, p], piVals[0, p] = getPiandLambda(
                    xpos[p], ypos[p], Lambda[t, :, :], Pi[t, :, :])

        # plt.pcolormesh(u[t,:,:], vmin = -1.5, vmax = 1.5)
        # plt.colorbar()
        # plt.show()
        xpos, ypos, LambdaVal, PiVal = updatePositon_linVel(xpos, ypos,
                                                            LambdaVal, PiVal,
                                                            Lambda[t+stride, :,
                                                                :], Pi[t+stride, :, :],
                                                            u[t, :, :], v[t, :, :],
                                                            u[t+stride, :, :], v[t+stride, :, :])
        if t < (timelen-stride):
            dummy = np.zeros((1, nParticles, ), dtype=np.float64)
            dummy[0,:] = xpos[:].copy()
            xposVals = np.concatenate((xposVals, dummy), axis=0)
            dummy[0,:] = ypos[:].copy()
            yposVals = np.concatenate((yposVals, dummy), axis=0)

            dummy[0, :] = LambdaVal[:].copy()
            lambdaVals =  np.concatenate((lambdaVals, dummy), axis=0)
            dummy[0, :] = PiVal[:].copy()
            piVals = np.concatenate((piVals, dummy), axis=0)

            cdfTimeVals.append(timeVal[t+stride])
        else:
            xpos_next[:] = xpos[:]
            ypos_next[:] = ypos[:]


writeDS = Dataset(wFname, 'w', format='NETCDF4_CLASSIC')
writeDS.createDimension('Time', None)
writeDS.createDimension('PID', nParticles)

wCDF_Time = writeDS.createVariable('Time', np.float64, ('Time'))
wCDF_Time.units = timeUnits

wCDF_xpos = writeDS.createVariable('xpos', np.float64, ('Time', 'PID'))
wCDF_ypos = writeDS.createVariable('ypos', np.float64, ('Time', 'PID'))

wCDF_nxpos = writeDS.createVariable('xpos_next', np.float64, ('PID'))
wCDF_nypos = writeDS.createVariable('ypos_next', np.float64, ('PID'))


wCDF_Lambda = writeDS.createVariable('Lambda', np.float64, ('Time', 'PID'))
wCDF_Pi = writeDS.createVariable('Pi', np.float64, ('Time', 'PID'))

(tlen, dummy) = np.shape(xposVals)
wCDF_Time[0:tlen] = np.array(cdfTimeVals)[:]
wCDF_xpos[:, :] = xposVals[:, :]
wCDF_ypos[:, :] = yposVals[:, :]

wCDF_nxpos[:] = xpos_next[:]
wCDF_nypos[:] = ypos_next[:]

wCDF_Lambda[:, :] = lambdaVals[:, :]
wCDF_Pi[:, :] = piVals[:, :]

writeDS.close()
