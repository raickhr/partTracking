from os import write
import numpy as np
import argparse
from netCDF4 import Dataset
import sys
from pathlib import Path

def getIndices(cur_xpos, cur_ypos, xh, yh):
    xlen = len(xh)
    ylen = len(yh)

    outDomain = False
    index_X = -999
    index_Y = -999

    if cur_xpos != cur_xpos or \
       cur_ypos != cur_ypos:

        outDomain = True
        index_X = -999
        index_Y = -999

        return index_X, index_Y, outDomain


    ## Out of domain
    if (cur_ypos >= yh[ylen-1]) or \
        (cur_ypos <= yh[0]) or \
        (abs(cur_xpos) > 1e5) or \
        (abs(cur_ypos) > 1e5):

        outDomain = True
        index_X = -999
        index_Y = -999

    else:
        try:
            index_X = int((cur_xpos - xh[0])//dx)
            index_Y = int((cur_ypos - yh[0])//dy)
            outDomain = False
        except:
            print('error in getting index', cur_xpos, cur_ypos)
            sys.exit()
        
        if index_X >= xlen - 1 or index_X < 0:
            print("error in x index, x_index", index_X, "cur_xpos", cur_xpos)
            print("x domain size: ", xh[0], xh[xlen-1])
            sys.exit()
        if index_Y >= ylen -1 or index_Y < 0:
            print("error in y index, y_index", index_Y, "cur_ypos", cur_ypos)
            print("y domain size: ", yh[0], yh[ylen-1])
            sys.exit()


    return index_X, index_Y, outDomain

def updateCyclicXdirection(cur_xpos, xh):
    xlen = len(xh)
    threshold = 0.0001
    if (cur_xpos >= xh[xlen-1]):
        #print('cur_xpos', cur_xpos)
        cur_xpos = (cur_xpos - xh[xlen-1]) + xh[0]
        #print('after cur_xpos', cur_xpos)
        if (cur_xpos - xh[0]) <= threshold:
            cur_xpos = xh[0]

    if (cur_xpos < xh[0]):
        #print('cur_xpos', cur_xpos)
        cur_xpos = xh[xlen-1] - (xh[0] - cur_xpos)
        #print('after cur_xpos', cur_xpos)
        if (xh[xlen-1] - cur_xpos) <= threshold:
            cur_xpos = xh[0]
        
    return cur_xpos

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

def getPiandLambda(index_X, index_Y, xpos, ypos, xh, yh, currentLambda, currentPi):
    ## Also works for RV and PV, Omega_tilde and S_tilde_sq, R_Barocl and S_baro
    ### need to give lower left corner index
    xlen = len(xh)
    ylen = len(yh)

    lambdaVal = -1e34
    PiVal = -1e34

    if index_X < 0 or index_X >= xlen-1 or index_Y < 0 or index_Y >= ylen -1:
        lambdaVal = -1e34
        PiVal = -1e34
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
            lambda_1 = currentLambda[y1, x1]
            lambda_2 = currentLambda[y2, x2]
            lambda_3 = currentLambda[y3, x3]
            lambda_4 = currentLambda[y4, x4]

            Pi_1 = currentPi[y1, x1]
            Pi_2 = currentPi[y2, x2]
            Pi_3 = currentPi[y3, x3]
            Pi_4 = currentPi[y4, x4]

        except:
            print("index error")
            sys.exit()
        
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


parser = argparse.ArgumentParser()

parser.add_argument("--nParticles", "-n", type=int, default=500, action='store',
                    help="number of particles to be tracked")
parser.add_argument("--file","-f", type=str, action='store', required=True,
                    help="file from where the particles will be tracked")
parser.add_argument("--filteredFile","-ff", type=str, action='store', required=True,
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
nextFilteredFile = fldLoc + '/prog_{0:03d}_LambdaAndPiValues.nc'.format(nextFileNum)

wFname = fldLoc + '/prog_{0:03d}_pTracking_{1:03d}p.nc'.format(fileNum, nParticles)

prevFileNum = fileNum - 1
prevFile = fldLoc + '/prog_{0:03d}_pTracking_{1:03d}p.nc'.format(prevFileNum, nParticles)
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
dt = (timeVal[1] - timeVal[0])#*24*3600 
substeps = 20

Lambda = np.array(ds2.variables['Lambda'])
Pi = np.array(ds2.variables['Pi'])

RV = np.array(ds.variables['RV'])
PV = np.array(ds.variables['PV'])

Omega_tilde = np.array(ds2.variables['Omega_tilde'])
S_tilde_sq = np.array(ds2.variables['S_tilde_sq'])

R_barocl = np.array(ds2.variables['R_Barocl'])
S_baro = np.array(ds2.variables['S_Baro'])

Lambda_str = np.array(ds2.variables['Lambda_str'])
Lambda_rot = np.array(ds2.variables['Lambda_rot'])

u = np.array(ds.variables['u'])
v = np.array(ds.variables['v'])

## Next file is needed for the last time index
if nextFilePath.is_file():
    print('continuing file present')
    ds3 = Dataset(nextFile)
    ds4 = Dataset(nextFilteredFile)
    (s1, s2, s3)  = np.shape(u)
    
    u1 = np.zeros((1,s2,s3), dtype =float)
    v1 = u1.copy()
    
    Lambda1  = u1.copy()
    Pi1 = u1.copy()

    Lambda_str1 = u1.copy()
    Lambda_rot1 = u1.copy()

    RV1 = u1.copy()
    PV1 = u1.copy()

    Omega_tilde1 = u1.copy()
    S_tilde_sq1 = u1.copy()

    R_barocl1 = u1.copy()
    S_baro1 = u1.copy()

    
    ## need future u1 values for the last step
    u1[0,:,:] = np.array(ds3.variables['u'])[0,:,:]
    v1[0,:,:] = np.array(ds3.variables['v'])[0,:,:]

    RV1[0, :, :] = np.array(ds3.variables['RV'][0, :, :])
    PV1[0, :, :] = np.array(ds3.variables['PV'][0, :, :])

    Lambda1[0, :, :] = np.array(ds4.variables['Lambda'][0, :, :])
    Pi1[0, :, :] = np.array(ds4.variables['Pi'][0, :, :])
    
    Lambda_str1[0, :, :] = np.array(ds4.variables['Lambda_str'][0, :, :])
    Lambda_rot1[0, :, :] = np.array(ds4.variables['Lambda_rot'][0, :, :])

    Omega_tilde1[0, :, :] = np.array(ds4.variables['Omega_tilde'][0, :, :])
    S_tilde_sq1[0, :, :] = np.array(ds4.variables['S_tilde_sq'][0, :, :])

    R_barocl1[0, :, :] = np.array(ds4.variables['R_Barocl'][0, :, :])
    S_baro1[0, :, :] = np.array(ds4.variables['S_Baro'][0, :, :])

    u = np.concatenate((u,u1), axis=0)
    v = np.concatenate((v,v1), axis=0)

    Lambda = np.concatenate((Lambda, Lambda1), axis=0)
    Pi = np.concatenate((Pi, Pi1), axis=0)

    RV = np.concatenate((RV, RV1), axis=0)
    PV = np.concatenate((PV, PV1), axis=0)

    Omega_tilde = np.concatenate((Omega_tilde, Omega_tilde1), axis=0)
    S_tilde_sq = np.concatenate((S_tilde_sq, S_tilde_sq1), axis=0)
    
    R_barocl = np.concatenate((R_barocl, R_barocl1), axis=0)
    S_baro = np.concatenate((S_baro, S_baro1), axis=0)

    Lambda_rot = np.concatenate((Lambda_rot, Lambda_rot1), axis=0)
    Lambda_str = np.concatenate((Lambda_str, Lambda_str1), axis=0)

else:
    timelen -= 1


## set the starting xpos and ypos and its field values
xpos_prev = np.zeros((nParticles), dtype=float)
ypos_prev = np.zeros((nParticles), dtype=float)

Pi_Val_prev, Lambda_Val_prev = xpos_prev.copy(), ypos_prev.copy()
Omega_tilde_Val_prev, S_tilde_sq_Val_prev = xpos_prev.copy(), ypos_prev.copy()
RV_Val_prev, PV_Val_prev = xpos_prev.copy(), ypos_prev.copy()
Lambda_str_Val_prev, Lambda_Rot_Val_prev = xpos_prev.copy(), ypos_prev.copy()
S_baro_Val_prev, R_barocl_Val_prev = xpos_prev.copy(), ypos_prev.copy()

## if previous file is present use the values from the previous file else generate random points and get values
if prevFilePath.is_file():
    print('getting positions from '+prevFile)
    ds5 = Dataset(prevFile)
    
    xpos_prev = np.array(ds5.variables['xpos_next'])
    ypos_prev = np.array(ds5.variables['ypos_next'])

    Pi_Val_prev = np.array(ds5.variables['Pi_next'])
    Lambda_Val_prev = np.array(ds5.variables['Lambda_next'])

    Lambda_str_Val_prev = np.array(ds5.variables['Lambda_str_next'])
    Lambda_rot_Val_prev = np.array(ds5.variables['Lambda_rot_next'])

    Omega_tilde_Val_prev = np.array(ds5.variables['Omega_tilde_next'])
    S_tilde_sq_Val_prev = np.array(ds5.variables['S_tilde_sq_next'])

    RV_Val_prev = np.array(ds5.variables['RV_next'])
    PV_Val_prev = np.array(ds5.variables['PV_next'])

    S_baro_Val_prev = np.array(ds5.variables['S_Baro_next'])
    R_barocl_Val_prev = np.array(ds5.variables['R_Barocl_next'])


else:
    xpos_prev = np.random.rand(nParticles) * (xh[xlen-1] - xh[0]) + xh[0]
    ypos_prev = np.random.rand(nParticles) * (yh[ylen-1] - yh[0]) + yh[0]

    for pid in range(nParticles):
        xpos_prev[pid] = updateCyclicXdirection(xpos_prev[pid], xh)
        index_X, index_Y, outDomain = getIndices(xpos_prev[pid], ypos_prev[pid], xh, yh)

        ### cyclic in X dirction
        
        if ~outDomain:
            Pi_Val_prev[pid], Lambda_Val_prev[pid] = getPiandLambda(
                index_X, index_Y, xpos_prev[pid], ypos_prev[pid], xh, yh, Lambda[0, :, :], Pi[0, :, :])
            
            RV_Val_prev[pid], PV_Val_prev[pid] = getPiandLambda(
                index_X, index_Y, xpos_prev[pid], ypos_prev[pid], xh, yh, RV[0, :, :], PV[0, :, :])

            Omega_tilde_Val_prev[pid], S_tilde_sq_Val_prev[pid] = getPiandLambda(
                index_X, index_Y, xpos_prev[pid], ypos_prev[pid], xh, yh, Omega_tilde[0, :, :], S_tilde_sq[0, :, :])
            
            Lambda_str_Val_prev[pid], Lambda_Rot_Val_prev[pid] = getPiandLambda(
                index_X, index_Y, xpos_prev[pid], ypos_prev[pid], xh, yh, Lambda_str[0, :, :], Lambda_rot[0, :, :])
            
            S_baro_Val_prev[pid], R_barocl_Val_prev[pid] = getPiandLambda(
                index_X, index_Y, xpos_prev[pid], ypos_prev[pid], xh, yh, S_baro[0, :, :], R_barocl[0, :, :])

        else:
            Pi_Val_prev[pid], Lambda_Val_prev[pid] = float('nan'), float('nan')
            RV_Val_prev[pid], PV_Val_prev[pid] = float('nan'), float('nan')
            Omega_tilde_Val_prev[pid], S_tilde_sq_Val_prev[pid] = float('nan'), float('nan')
            Lambda_str_Val_prev[pid], Lambda_Rot_Val_prev[pid] = float('nan'), float('nan')
            S_baro_Val_prev[pid], R_barocl_Val_prev[pid] = float('nan'), float('nan')
    
def updatePositon_linVel(xpos, ypos,
                         currentLambda, currentPi,
                         currentRV, currentPV,
                         currentOmega_tilde, currentS_tilde_sq,
                         currentLambda_str, currentLambda_rot,
                         currentR_Barocl, currentS_baro,
                         u1, v1, u2, v2):

    global dt, substeps, xh, yh

    Lambda_Val = np.zeros((nParticles), dtype=np.float32)
    Pi_Val = np.zeros((nParticles), dtype=np.float32)

    RV_Val = np.zeros((nParticles), dtype=np.float32)
    PV_Val = np.zeros((nParticles), dtype=np.float32)

    Omega_tilde_Val = np.zeros((nParticles), dtype=np.float32)
    S_tilde_sq_Val = np.zeros((nParticles), dtype=np.float32)

    Lambda_str_Val = np.zeros((nParticles), dtype=np.float32)
    Lambda_rot_Val = np.zeros((nParticles), dtype=np.float32)

    R_barocl_Val = np.zeros((nParticles), dtype=np.float32)
    S_baro_Val = np.zeros((nParticles), dtype=np.float32)

    continueLoop = np.zeros((nParticles), dtype=bool)

    for i in range(nParticles):
        if continueLoop[i]:
            continue
        cur_xpos = xpos[i]
        cur_ypos = ypos[i]
        subtimeStepLoopBroken = False
        for j in range(substeps):
            cur_xpos = updateCyclicXdirection(cur_xpos, xh)
            index_X, index_Y, continueLoop[i] = getIndices(cur_xpos, cur_ypos, xh, yh)

            if continueLoop[i]:
                subtimeStepLoopBroken = True 
                break
            ## x-displacement values at all the four corners
            #weight according to time
            w2 = j/(substeps-1)
            w1 = 1 - w2
            
            part_xdisp1 = ((w1 * u1[index_Y, index_X]) + (w2 * u2[index_Y, index_X]) ) * dt/substeps 
            part_xdisp2 = ((w1 * u1[index_Y, index_X+1]) + (w2 * u2[index_Y, index_X+1]) ) * dt/substeps 
            part_xdisp3 = ((w1 * u1[index_Y+1, index_X+1]) + (w2 * u2[index_Y+1, index_X+1]) ) * dt/substeps 
            part_xdisp4 = ((w1 * u1[index_Y+1, index_X]) + (w2 * u2[index_Y+1, index_X]) ) * dt/substeps 

            part_ydisp1 = ((w1 * v1[index_Y, index_X]) + (w2 * v2[index_Y, index_X]) ) * dt/substeps 
            part_ydisp2 = ((w1 * v1[index_Y, index_X+1]) + (w2 * v2[index_Y, index_X+1]) ) * dt/substeps 
            part_ydisp3 = ((w1 * v1[index_Y+1, index_X+1]) + (w2 * v2[index_Y+1, index_X+1]) ) * dt/substeps 
            part_ydisp4 = ((w1 * v1[index_Y+1, index_X]) + (w2 * v2[index_Y+1, index_X]) ) * dt/substeps 


            part_xdisp = interpolate(part_xdisp1,
                                    part_xdisp2,
                                    part_xdisp3,
                                    part_xdisp4,
                                    xpos[i] - xh[index_X], ypos[i] - yh[index_Y], dx, dy)

            part_ydisp = interpolate(part_ydisp1,
                                    part_ydisp2,
                                    part_ydisp3,
                                    part_ydisp4,
                                    xpos[i] - xh[index_X], ypos[i] - yh[index_Y], dx, dy)

            if part_xdisp == 0.0 and part_ydisp == 0.0:
                print('Displacement = 0 for particle', i)
            cur_xpos += part_xdisp/1000
            cur_ypos += part_ydisp/1000

        xpos[i] = cur_xpos
        ypos[i] = cur_ypos

        cur_xpos = updateCyclicXdirection(cur_xpos, xh)
        index_X, index_Y, continueLoop[i] = getIndices(
            cur_xpos, cur_ypos, xh, yh)

        if continueLoop[i] or subtimeStepLoopBroken:
            xpos[i], ypos[i] = float('nan'), float('nan')

            Lambda_Val[i], Pi_Val[i] = float('nan'), float('nan')
            RV_Val[i], PV_Val[i] = float('nan'), float('nan')

            Omega_tilde_Val[i], S_tilde_sq_Val[i]  = float('nan'), float('nan')
            S_baro_Val[i], R_barocl_Val[i] = float('nan'), float('nan')

            Lambda_str_Val[i], Lambda_rot_Val[i] = float('nan'), float('nan')

        else:
            Lambda_Val[i], Pi_Val[i] = getPiandLambda(
                index_X, index_Y, xpos[i], ypos[i], xh, yh, currentLambda, currentPi)

            RV_Val[i], PV_Val[i] = getPiandLambda(
                index_X, index_Y, xpos[i], ypos[i], xh, yh, currentRV, currentPV)

            Omega_tilde_Val[i], S_tilde_sq_Val[i] = getPiandLambda(
                index_X, index_Y, xpos[i], ypos[i], xh, yh, currentOmega_tilde, currentS_tilde_sq)

            S_baro_Val[i], R_barocl_Val[i] = getPiandLambda(
                index_X, index_Y, xpos[i], ypos[i], xh, yh, currentS_baro, currentR_Barocl)

            Lambda_str_Val[i], Lambda_rot_Val[i] = getPiandLambda(
                index_X, index_Y, xpos[i], ypos[i],  xh, yh, currentLambda_str, currentLambda_rot)

            
    return(xpos, ypos,
           Lambda_Val, Pi_Val,
           Lambda_rot_Val, Lambda_str_Val,
           RV_Val, PV_Val,
           Omega_tilde_Val, S_tilde_sq_Val,
           S_baro_Val, R_barocl_Val)


xpos_Vals = np.zeros((timelen, nParticles, ), dtype=float)
ypos_Vals = np.zeros((timelen, nParticles, ), dtype=float)
xpos_next = np.zeros((nParticles, ), dtype=float)
ypos_next = np.zeros((nParticles, ), dtype=float)

pi_Vals = np.zeros((timelen, nParticles, ), dtype=float)
lambda_Vals = np.zeros((timelen, nParticles, ), dtype=float)
pi_next = np.zeros((nParticles ), dtype=float)
lambda_next = np.zeros((nParticles ), dtype=float)

rv_Vals = np.zeros((timelen, nParticles, ), dtype=float)
pv_Vals = np.zeros((timelen, nParticles, ), dtype=float)
rv_next = np.zeros((nParticles ), dtype=float)
pv_next = np.zeros((nParticles ), dtype=float)

omega_tilde_Vals = np.zeros((timelen, nParticles, ), dtype=float)
s_tilde_sq_Vals = np.zeros((timelen, nParticles, ), dtype=float)
omega_tilde_next = np.zeros((nParticles ), dtype=float)
s_tilde_sq_next = np.zeros((nParticles ), dtype=float)

lambda_str_Vals = np.zeros((timelen, nParticles, ), dtype=float)
lambda_rot_Vals = np.zeros((timelen, nParticles, ), dtype=float)
lambda_str_next = np.zeros((nParticles ), dtype=float)
lambda_rot_next = np.zeros((nParticles ), dtype=float)

r_barocl_Vals = np.zeros((timelen, nParticles, ), dtype=float)
s_baro_Vals = np.zeros((timelen, nParticles, ), dtype=float)
r_barocl_next = np.zeros((nParticles ), dtype=float)
s_baro_next = np.zeros((nParticles ), dtype=float)

for t in range(0,timelen):
    print('time :', t)
    if t == 0:
        xpos_Vals[0, :], ypos_Vals[0,:] = xpos_prev, ypos_prev
        lambda_Vals[0, :], pi_Vals[0, :] = Lambda_Val_prev, Pi_Val_prev
        rv_Vals[0, :], pv_Vals[0, :] = RV_Val_prev, PV_Val_prev
        omega_tilde_Vals[0, :], s_tilde_sq_Vals[0, :] = Omega_tilde_Val_prev, S_tilde_sq_Val_prev
        r_barocl_Vals[0,:], s_baro_Vals[0, :] = R_barocl_Val_prev, S_baro_Val_prev

    xpos, ypos, Lambda_Val, Pi_Val, Lambda_rot_Val, Lambda_str_Val, \
    RV_Val, PV_Val, Omega_tilde_Val, S_tilde_sq_Val, \
    S_baro_Val, R_barocl_Val = updatePositon_linVel(xpos_prev, ypos_prev,
                                                    Lambda[t+1,:,:], Pi[t+1,:,:],
                                                    RV[t+1,:,:], PV[t+1,:,:],
                                                    Omega_tilde[t+1,:,:], S_tilde_sq[t+1,:,:],
                                                    Lambda_str[t+1,:,:], Lambda_rot[t+1,:,:],
                                                    R_barocl[t+1,:,:], S_baro[t+1,:,:],
                                                    u[t,:,:], v[t,:,:], u[t+1,:,:], v[t+1,:,:])
            
    if t < (timelen-1):
        xpos_Vals[t+1, :], ypos_Vals[t+1, :] = xpos[:], ypos[:]
        lambda_Vals[t+1, :], pi_Vals[t+1, :] = Lambda_Val[:], Pi_Val[:]
        lambda_str_Vals[t+1, :], lambda_rot_Vals[t+1, :] = Lambda_str_Val[:], Lambda_rot_Val[:]
        omega_tilde_Vals[t+1, :], s_tilde_sq_Vals[t+1, :] = Omega_tilde_Val[:], S_tilde_sq_Val[:]
        r_barocl_Vals[t+1, :], s_baro_Vals[t+1, :] = R_barocl_Val[:], S_baro_Val[:]
        rv_Vals[t+1, :], pv_Vals[t+1, :] = RV_Val[:], PV_Val[:]

    elif t == (timelen-1):
        xpos_next[:], ypos_next[:] = xpos[:], ypos[:]
        lambda_next[:], pi_next[:] = Lambda_Val[:], Pi_Val[:]
        lambda_str_next[:], lambda_rot_next[:] = Lambda_str_Val[:], Lambda_rot_Val[:]
        omega_tilde_next[:], s_tilde_sq_next[:] = Omega_tilde_Val[:], S_tilde_sq_Val[:]
        r_barocl_next[:], s_baro_next[:] = R_barocl_Val[:], S_baro_Val[:]
        rv_next[:], pv_next[:] = RV_Val[:], PV_Val[:]

    xpos_prev, ypos_prev = xpos, ypos



writeDS = Dataset(wFname, 'w', format='NETCDF4_CLASSIC')
writeDS.createDimension('Time', None)
writeDS.createDimension('PID', nParticles)

wCDF_Time = writeDS.createVariable('Time', np.float32, ('Time'))
wCDF_Time.units = timeUnits

wCDF_xpos = writeDS.createVariable('xpos', np.float32, ('Time','PID'))
wCDF_ypos = writeDS.createVariable('ypos', np.float32, ('Time', 'PID'))
wCDF_nxpos = writeDS.createVariable('xpos_next', np.float32, ('PID'))
wCDF_nypos = writeDS.createVariable('ypos_next', np.float32, ('PID'))


wCDF_Lambda = writeDS.createVariable('Lambda', np.float32, ('Time', 'PID'))
wCDF_Pi = writeDS.createVariable('Pi', np.float32, ('Time', 'PID'))
wCDF_Lambda_next = writeDS.createVariable('Lambda_next', np.float32, ('PID'))
wCDF_Pi_next = writeDS.createVariable('Pi_next', np.float32, ('PID'))


wCDF_RV = writeDS.createVariable('RV', np.float32, ('Time', 'PID'))
wCDF_PV = writeDS.createVariable('PV', np.float32, ('Time', 'PID'))
wCDF_RV_next = writeDS.createVariable('RV_next', np.float32, ('PID'))
wCDF_PV_next = writeDS.createVariable('PV_next', np.float32, ('PID'))

wCDF_Omega_tilde = writeDS.createVariable('Omega_tilde', np.float32, ('Time', 'PID'))
wCDF_S_tilde_sq = writeDS.createVariable('S_tilde_sq', np.float32, ('Time', 'PID'))
wCDF_Omega_tilde_next = writeDS.createVariable('Omega_tilde_next', np.float32, ('PID'))
wCDF_S_tilde_sq_next = writeDS.createVariable('S_tilde_sq_next', np.float32, ('PID'))

wCDF_Lambda_str = writeDS.createVariable('Lambda_str', np.float32, ('Time', 'PID'))
wCDF_Lambda_rot = writeDS.createVariable('Lambda_rot', np.float32, ('Time', 'PID'))
wCDF_Lambda_str_next = writeDS.createVariable('Lambda_str_next', np.float32, ('PID'))
wCDF_Lambda_rot_next = writeDS.createVariable('Lambda_rot_next', np.float32, ('PID'))

wCDF_R_barocl = writeDS.createVariable('R_Barocl', np.float32, ('Time', 'PID'))
wCDF_S_baro = writeDS.createVariable('S_Baro', np.float32, ('Time', 'PID'))
wCDF_R_barocl_next = writeDS.createVariable('R_Barocl_next', np.float32, ('PID'))
wCDF_S_baro_next = writeDS.createVariable('S_Baro_next', np.float32, ('PID'))


wCDF_Time[0:timelen] = timeVal[0:timelen]

wCDF_xpos[:,:] = xpos_Vals[:,:]
wCDF_ypos[:,:] = ypos_Vals[:,:]
wCDF_nxpos[:] = xpos_next[:]
wCDF_nypos[:] = ypos_next[:]

wCDF_Lambda[:,:] = lambda_Vals[:,:]
wCDF_Pi[:,:] = pi_Vals[:,:]
wCDF_Lambda_next[:] = lambda_next[:]
wCDF_Pi_next[:] = pi_next[:]

wCDF_RV[:, :] = rv_Vals[:, :]
wCDF_PV[:, :] = pv_Vals[:, :]
wCDF_RV_next[:] = rv_next[:]
wCDF_PV_next[:] = pv_next[:]

wCDF_Lambda_str[:, :] = lambda_str_Vals[:, :]
wCDF_Lambda_rot[:, :] = lambda_rot_Vals[:, :]
wCDF_Lambda_str_next[:] = lambda_str_next[:]
wCDF_Lambda_rot_next[:] = lambda_rot_next[:]

wCDF_Omega_tilde[:, :] = omega_tilde_Vals[:, :]
wCDF_S_tilde_sq[:, :] = s_tilde_sq_Vals[:, :]
wCDF_Omega_tilde_next[:] = omega_tilde_next[:]
wCDF_S_tilde_sq_next[:] = s_tilde_sq_next[:]
 
wCDF_R_barocl[:, :] = r_barocl_Vals[:, :]
wCDF_S_baro[:, :] = s_baro_Vals[:, :]
wCDF_R_barocl_next[:] = r_barocl_next[:]
wCDF_S_baro_next[:] = s_baro_next[:]

writeDS.close()





