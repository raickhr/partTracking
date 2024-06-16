import numpy as np
import sys
from netCDF4 import Dataset


def biLinearInterpolateSinglePoint(x, y, westEastCoord , southNorthCoord, val2DArr):
    #westEastCoord  shape (2)
    #southNorthCoord shape (2)
    #ValArr shape (2,2)

    #ValArr are values at the corner
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
    xArr = np.array([westEastCoord[0], westEastCoord[1], westEastCoord[1], westEastCoord[0]], dtype=np.float64)
    yArr = np.array([southNorthCoord[0], southNorthCoord[0], southNorthCoord[1], southNorthCoord[1]], dtype=np.float64)
    ValArr = np.array([val2DArr[0,0], val2DArr[1,0], val2DArr[1,1], val2DArr[0,1]], dtype=np.float64)

    xyArr = xArr * yArr
    Amat = np.stack((np.ones((4),dtype=float),
                    xArr,
                    yArr,
                    xyArr), axis =1)
    
    coeff = np.linalg.inv(Amat)@ValArr
    multArr = np.array([1, x, y, x*y], dtype=float)
    return coeff@multArr

def vertLinearInterpolationSinglePoint(z, zArr, ValArr):
    dVal = ValArr[1] -ValArr[0] 
    val = (z-zArr[0])/(zArr[1] - zArr[0]) * dVal + ValArr[0]
    return val

def getNearestIndices(x, y, z, xcoord, ycoord, zcoord):
    # This function returns indices of the points around the positions given by x and y values
    npoints = len(x)
    if len(y) != npoints or len(z) != npoints:
        print('xlen is not equal to ylen or zlen')
        sys.exit()
    
    Xindices = np.zeros((npoints,2), dtype=int)
    Yindices = np.zeros((npoints,2), dtype=int)
    Zindices = np.zeros((npoints,2), dtype=int)

    for i in range(npoints):
        westInd = 0
        eastInd = 0

        westInd = np.argmin(abs(x[i]-xcoord))
        if xcoord[westInd] < x[i]:
            eastInd = westInd+1
        else:
            eastInd = westInd
            westInd = eastInd-1

        Xindices[i,0] = westInd
        Xindices[i,1] = eastInd

        ###############################

        southInd = 0
        northInd = 0

        southInd = np.argmin(abs(y[i]-ycoord))
        if ycoord[southInd] < y[i]:
            northInd = southInd+1
        else:
            northInd = southInd
            southInd = northInd-1

        Yindices[i,0] = southInd
        Yindices[i,1] = northInd

        ###############################
        
        bottomInd = 0
        topInd = 0

        bottomInd = np.argmin(abs(z[i]-zcoord))
        if zcoord[bottomInd] < z[i]:
            topInd = bottomInd+1
        else:
            topInd = bottomInd
            bottomInd = topInd-1

        Zindices[i,0] = bottomInd
        Zindices[i,1] = topInd

    return Xindices, Yindices, Zindices


def readnetCDFbyIndices(file, Xindices, Yindices, Zindices, varList):
    ## Xindices shape (number of points, 2) the 2 indices refer to 0 -> west, 1-> east
    ## Yindices shape (number of points, 2) the 2 indices refer to 0 -> south, 1-> north
    ## Zindices shape (number of points, 2) the 2 indices refer to 0 -> bottom, 1-> top
    ## returns arr of shape (nvars, npoints, 2, 2)

    nvars = len(varList)
    npoints = Xindices.shape[0]
    
    arr = np.zeros((nvars, npoints, 2, 2, 2), dtype=float)

    ds = Dataset(file)

    for i in range(nvars):
        varName = varList[i]
        for pindx in range(npoints):
            xind = Xindices[pindx,:]
            yind = Yindices[pindx,:]
            zind = Zindices[pindx,:]
            #print('zind,yind,xind',zind,yind,xind)
            arr[i,pindx,:,:,:] = np.array(ds.variables[varName][0,zind,yind,xind], dtype=np.float64)
            #print('pindx', varName,'\n', arr[i,pindx,:,:,:])
    ds.close()
    return arr

def getVarsAtPos(xpos, ypos, zpos, xcoord, ycoord, zcoord, file, varList):
    #returns variable array of shape (nvars, npoints)
    # xpos, ypos and zpos are of shape (npoints)

    npoints = len(xpos)
    nvars = len(varList)

    Xindices, Yindices, Zindices = getNearestIndices(xpos, ypos, zpos, xcoord, ycoord, zcoord)
    # Xindices, Yindices, Zindices all of shape (npoints,2)
    #print('xpos, ypos, zpos',xpos, ypos, zpos)
    #print('Xindices, Yindices, Zindices', Xindices, Yindices, Zindices)
    
    
    maskx = np.logical_or(Xindices >= len(xcoord) , Xindices < 0)
    masky = np.logical_or(Yindices >= len(ycoord) , Yindices < 0)
    maskz = Zindices >= len(zcoord)

    maskx = np.sum(maskx, axis=1)
    masky = np.sum(masky, axis=1)
    maskz = np.sum(maskz, axis=1)

    maskParticle = np.array(maskx + masky + maskz, dtype=bool)
    #print('maskParticle', maskParticle)
    Xindices[maskParticle,0] = 0
    Xindices[maskParticle,1] = 1
    Yindices[maskParticle,0] = 0
    Yindices[maskParticle,1] = 1
    Zindices[maskParticle,0] = 0
    Zindices[maskParticle,1] = 1

    maskz = Zindices < 0
    Zindices[maskz] = 0
    #print('Zindices', Zindices[0,:])




    xcoordVals = xcoord[Xindices.flatten()]
    ycoordVals = ycoord[Yindices.flatten()]
    zcoordVals = zcoord[Zindices.flatten()]

    xcoordVals = xcoordVals.reshape(npoints,2)
    ycoordVals = ycoordVals.reshape(npoints,2)
    zcoordVals = zcoordVals.reshape(npoints,2)

    varValsAtCorners = readnetCDFbyIndices(file, Xindices, Yindices, Zindices, varList)
    ## returns arr of shape (nvars, npoints, 2, 2, 2) 2 in vertical dir, 2 in south-north, 2 in west-east

    varVals = np.zeros((nvars,npoints))

    for i in range(nvars):
        for j in range(npoints):
            ### remember bottom and top are opposite because z co-ordinates is depth co-ordinates
            # if i < 3 and j<1:
            #     print(f'before interpolatoin var {varList[i]} \n', varValsAtCorners[i,j,0,:,:])
            #     print(f'interpolation coords:')
            #     print('xpos ypos', xpos[j], ypos[j])
            #     print('xcorners', xcoordVals[j,:])
            #     print('ycorners', ycoordVals[j,:])
            bottomArr = biLinearInterpolateSinglePoint(xpos[j], ypos[j], 
                                                       xcoordVals[j,:] , 
                                                       ycoordVals[j,:], 
                                                       varValsAtCorners[i,j,0,:,:])
            # if i < 3 and j<1:
            #     print(f'after interpolation var {varList[i]}', bottomArr, '\n')
            #     print(f'bottom var {varList[i]}', bottomArr)
            topArr = biLinearInterpolateSinglePoint(xpos[j], 
                                                    ypos[j], 
                                                    xcoordVals[j,:] , 
                                                    ycoordVals[j,:], 
                                                    varValsAtCorners[i,j,1,:,:])
            # if i <3 and j<1:
            #     print(f'top var {varList[i]}', topArr)
            #     print(f'interpolation coords:')
            #     print('zpos', zpos[j])
            #     print('zcorners', zcoordVals[j,:])
            varVals[i,j] = vertLinearInterpolationSinglePoint(zpos[j], 
                                                              zcoordVals[j,:], 
                                                              np.array([bottomArr, topArr]))
            
            # if i<3 and j<1:
            #     print(f'after vertical interpolation var {varList[i]}', varVals[i,j], '\n')

    varVals[:,maskParticle] = np.nan
    return varVals
            

def updatePositions(dt, xpos, ypos, zpos,  xvel, yvel, zvel):
    earthRad = 6.371e6
    lats = np.deg2rad(ypos)
    R = earthRad * np.cos(lats)

    dx = np.rad2deg((xvel * dt.seconds)/R)
    dy = np.rad2deg((yvel * dt.seconds)/earthRad)
    dz = -zvel * dt.seconds
    # print('In update Positions')
    # print(f'xvel, yvel, zvel = {xvel[0]}, {yvel[0]}, {zvel[0]}')
    print(f'dx, dy, dz = {dx[0]}, {dy[0]}, {dz[0]}')
    # print(f'xpos, ypos, zpos = {xpos[0]}, {ypos[0]}, {zpos[0]}')
    # print('\n')
    xpos += dx
    ypos += dy
    zpos += dz

    return xpos, ypos, zpos




        