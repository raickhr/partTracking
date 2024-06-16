import numpy as np
from interpolatoinFunctions import *
from writeParticleFile import *
from configuration import *

import warnings
warnings.filterwarnings("ignore")

###################

config = configClass()

startFile = config.startFile

timeVal = config.getTimeVal(config.readFolder + startFile)
timeList = config.getTimeList()

xcoord, ycoord, zcoord = config.getCoordinates()
varNameList = config.varNameList
print(f'number of variables is {len(varNameList)}')
varUnitsList = config.varUnitsList

partFileName = config.writeFolder + config.writeFileName
nsubTimeSteps = config.nsubTimeSteps

readFolder = config.readFolder
writeFolder = config.writeFolder

xvelVarIndx = config.xvelVarIndx
yvelVarIndx = config.yvelVarIndx
zvelVarIndx = config.zvelVarIndx
cartZvarIndx = config.cartZvarIndx
xSigmaSlopeIndx = config.xSigmaSlopeIndx
ySigmaSlopeIndx = config.ySigmaSlopeIndx
thicknessPerSigmaIndx = config.thicknessPerSigmaIndx


xpos, ypos, zpos = config.getStartPartPos()
#print('start position for first particle', xpos[0], ypos[0], zpos[0])


file0 = config.readFolder + startFile
varValList = getVarsAtPos(xpos, ypos, zpos, xcoord, ycoord, zcoord,  cartZvarIndx, file0, varNameList)

xvel = varValList[xvelVarIndx,:]
yvel = varValList[yvelVarIndx,:]
zvel = varValList[zvelVarIndx,:]
dx_z = varValList[xSigmaSlopeIndx,:]
dy_z = varValList[ySigmaSlopeIndx,:]
dz_z = varValList[thicknessPerSigmaIndx,:]
#print(xvel.shape)
#print('xvel, yvel, zvel', xvel, yvel, zvel)

#sys.exit()

createParticleFile(np.array([xpos], dtype=np.float64), 
                   np.array([ypos], dtype=np.float64), 
                   np.array([zpos], dtype=np.float64), 
                   np.array([timeVal]), 
                   varNameList, 
                   varUnitsList, 
                   np.array([varValList], dtype=np.float64), 
                   partFileName)

###################

for timeIdx in range(len(timeList)-1):
    time = timeList[timeIdx]
    nextTime = timeList[timeIdx+1]
    file0 = readFolder + config.getFileName(time)
    file1 = readFolder + config.getFileName(nextTime)

    for i in range(nsubTimeSteps):
        tval0 = config.getTimeVal(file0)  
        varVals0 = getVarsAtPos(xpos, ypos, zpos, xcoord, ycoord, zcoord,  cartZvarIndx, file0, varNameList)
        #print(f'file0 uo = {varVals0[0,0]:8.4f}, vo = {varVals0[1,0]:8.4f}, wo = {varVals0[2,0]:8.4f}')

        tval1 = config.getTimeVal(file1)
        varVals1 = getVarsAtPos(xpos, ypos, zpos, xcoord, ycoord, zcoord,  cartZvarIndx, file1, varNameList)
        #print(f'file1 uo = {varVals1[0,0]:8.4f}, vo = {varVals1[1,0]:8.4f}, wo = {varVals1[2,0]:8.4f}')

        dt = (tval1-tval0)/nsubTimeSteps
        curTime = tval0 + dt*i
        print('at ',curTime)
        timeVal = [curTime] #np.array([curTime])

        ### linear interpolation in time##
        frac = (dt*i)/(tval1- tval0)
        varValsDiff = varVals1 - varVals0
        curVars = varVals0 + frac*varValsDiff
        #print(f'used uo = {curVars[0,0]}, vo = {curVars[1,0]}, wo = {curVars[2,0]}')
        ### linear interpolation in time##

        #print(f'dt = {dt}')
        #print('frac = ', (dt*i)/(tval1- tval0))
        xvel = curVars[xvelVarIndx,:]
        yvel = curVars[yvelVarIndx,:]
        zvel = curVars[zvelVarIndx,:]
        dxi_z = curVars[xSigmaSlopeIndx,:]
        deta_z = curVars[ySigmaSlopeIndx,:]
        dsigma_z = curVars[thicknessPerSigmaIndx,:]

        xpos, ypos, zpos =  updatePositions(dt, xpos, ypos, zpos,  xvel, yvel, zvel, dxi_z, deta_z, dsigma_z )

        appendParticleFile(np.array([xpos], dtype=np.float64), 
                   np.array([ypos], dtype=np.float64), 
                   np.array([zpos], dtype=np.float64), 
                   np.array(timeVal), 
                   varNameList, 
                   varUnitsList, 
                   np.array([curVars], dtype=np.float64), 
                   partFileName)
    #sys.exit()
###################



