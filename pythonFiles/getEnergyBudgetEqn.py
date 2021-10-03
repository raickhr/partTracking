from writeFilteredFields import S_tilde_11, U_tilde
from netCDF4 import Dataset
import numpy as np
from filteringFunctions import *
from lagrangianParticles import getIndices, updateCyclicXdirection, interpolate
import argparse
from pathlib import Path
import sys

dx = dy = 5000
dt = 900

def getInterpolatedField(xpos, ypos, xh, yh, FeildVal):
    #returns interpolated field value
    xpos = updateCyclicXdirection(xpos, xh)
    index_X, index_Y, outDomain = getIndices(xpos, ypos, xh, yh)

    if outDomain:
        return float('nan')
    
    xlen = len(xh)
    ylen = len(yh)

    dx = xh[1] - xh[0]
    dy = yh[1] - yh[0]

    returnFieldVal = -1e34

    if index_X < 0 or index_X >= xlen-1 or index_Y < 0 or index_Y >= ylen - 1:
        returnFieldVal = -1e34
    else:

        x1 = index_X
        y1 = index_Y

        x2 = index_X+1
        y2 = index_Y

        x3 = index_X+1
        y3 = index_Y+1

        x4 = index_X
        y4 = index_Y+1

        xpos = xpos - xh[index_X]
        ypos = ypos - yh[index_Y]

        try:
            Field_1 = FeildVal[y1, x1]
            Field_2 = FeildVal[y2, x2]
            Field_3 = FeildVal[y3, x3]
            Field_4 = FeildVal[y4, x4]

        except:
            print("index error")
            sys.exit()

        returnFieldVal = interpolate(Field_1,
                                Field_2,
                                Field_3,
                                Field_4,
                                xpos, ypos, dx, dy)

    return returnFieldVal


def getFeild(ds, fieldName, timeIndx, yindex, xindex):
    return np.array(ds.variables[fieldName][timeIndx, yindex, xindex])

def getFeild2D(ds, fieldName, timeIndx):
    return np.array(ds.variables[fieldName][timeIndx, :, :])

parser = argparse.ArgumentParser()

parser.add_argument("--feildsFile", "-ff", type=str, default='FilteredFileds.nc', action='store',
                    help="this is the filtered feilds file")

parser.add_argument("--particleFile", "-pf", type=str, default='particleFile', action='store',
                    help="this is the particle file")

parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")

args = parser.parse_args()

fldLoc = args.fldLoc

ffileName = args.feildsFile
ffds = Dataset(ffileName)
fftime = np.array(ffds.variables['Time'])
xh = np.array(ffds.variables['xh'])
yh = np.array(ffds.variables['yh'])
timeUnits = ffds.variables['Time'].units
timelen = len(fftime)

ffTimeStep = fftime[1] - fftime[0]

fileNum = int(ffileName.lstrip('prog_').rstrip('_LambdaAndPiValues.nc'))
nextFileNum = fileNum + 1
nextFile = fldLoc + '/prog_{0:03d}_LambdaAndPiValues.nc'.format(nextFileNum)
nextFilePath = Path(nextFile)
nextFilteredFile = fldLoc + \
    '/prog_{0:03d}_LambdaAndPiValues.nc'.format(nextFileNum)

nffds = None

if nextFilePath.is_file():
    nffds = Dataset(nextFile)
else:
    timelen -= 1

pfileName = args.particleFile
pfds = Dataset(pfileName)
pftime = np.array(pfds.variables['Time'])

xpos = pfds.variables['xpos']
ypos = pfds.variables['ypos']


nPtime, nParticles = np.shape(xpos)
DDt_Omega_Tilde_sq = np.zeros((nPtime, nParticles), dtype=float)
DDt_S_Tilde_sq = np.zeros((nPtime, nParticles), dtype=float)


for timeIndex in range(0,timelen):
    global xh, yh, ds, nds
    ndsBool = False
    Omega_TildeSq_t0 = getFeild2D(ds, 'Omega_tilde', timeIndex)**2
    Omega_TildeSq_t1 = Omega_TildeSq_t0.copy()
    
    S_TildeSq_t0 = getFeild2D(ds, 'S_tilde_sq', timeIndex)
    S_TildeSq_t1 = S_TildeSq_t0.copy()

    U_Tilde = getFeild2D(ds, 'u_tilde', timeIndex)
    V_Tilde = getFeild2D(ds, 'v_tilde', timeIndex)

    if timeIndex == 999:
        Omega_TildeSq_t1 = getFeild2D(nds, 'Omega_tilde', 0)**2
        S_TildeSq_t1 = getFeild2D(nds, 'S_tilde_sq', 0)


    ddx_Omega_TildeSq_t0, ddy_Omega_TildeSq_t0 = getGradient(Omega_TildeSq_t0, dx, dy)
    ddx_S_TildeSq_t0, ddy_S_TildeSq_t0 = getGradient(S_TildeSq_t0, dx, dy)

    for Pid in range(nParticles):
        cur_xpos, cur_ypos = xpos[timeIndex, Pid], ypos[timeIndex, Pid]
        
        omega_Tilde_sq_t0_xy = getInterpolatedField(xpos, ypos, xh, yh, Omega_TildeSq_t0)
        omega_Tilde_sq_t1_xy = getInterpolatedField(xpos, ypos, xh, yh, Omega_TildeSq_t1)

        s_Tilde_sq_t0_xy = getInterpolatedField(xpos, ypos, xh, yh, S_TildeSq_t0)
        s_Tilde_sq_t1_xy = getInterpolatedField(xpos, ypos, xh, yh, S_TildeSq_t1)

        ddx_Omega_TildeSq_xy = getInterpolatedField(xpos, ypos, xh, yh, ddx_Omega_TildeSq_t0)
        ddy_Omega_TildeSq_xy = getInterpolatedField(xpos, ypos, xh, yh, ddy_Omega_TildeSq_t0)

        ddx_S_TildeSq_xy = getInterpolatedField(xpos, ypos, xh, yh, ddx_S_TildeSq_t0)
        ddy_S_TildeSq_xy = getInterpolatedField(xpos, ypos, xh, yh, ddy_S_TildeSq_t0)

        u_Tilde_xy = getInterpolatedField(xpos, ypos, xh, yh, U_Tilde)
        v_Tilde_xy = getInterpolatedField(xpos, ypos, xh, yh, V_Tilde)

        DDt_Omega_Tilde_sq[timeIndex, Pid] = (omega_Tilde_sq_t1_xy - omega_Tilde_sq_t0_xy)/dt + \
            u_Tilde_xy * ddx_Omega_TildeSq_xy + \
            v_Tilde_xy * ddy_Omega_TildeSq_xy

        DDt_S_Tilde_sq[timeIndex, Pid] = (s_Tilde_sq_t1_xy - s_Tilde_sq_t0_xy)/dt + \
            u_Tilde_xy * ddx_S_TildeSq_xy + \
            v_Tilde_xy * ddy_S_TildeSq_xy
            
    
writeDS = Dataset(wFname, 'w', format='NETCDF4_CLASSIC')
writeDS.createDimension('Time', None)
writeDS.createDimension('PID', nParticles)

wCDF_Time = writeDS.createVariable('Time', np.float32, ('Time'))
wCDF_Time.units = timeUnits
wCDF_Time[0:timelen] = fftime[0:timelen]

wCDF_var1 = writeDS.createVariable('Ddt_Omega_TildeSq', np.float32, ('Time', 'PID'))
wCDF_var2 = writeDS.createVariable('Ddt_S_TildeSq', np.float32, ('Time', 'PID'))

wCDF_var1[:, :] = DDt_Omega_Tilde_sq[:,:]
wCDF_var2








