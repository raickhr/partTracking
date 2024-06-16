import numpy as np
import sys
from scipy.interpolate import LinearNDInterpolator
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
    # remember the depth coordinates is depth or height
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
            arr[i,pindx,:,:,:] = np.array(ds.variables[varName][0,zind,yind,xind], dtype=np.float64)
            # if varName == 'wo' :
            #     print('zind,yind,xind',zind,yind,xind)
            #     print('pindx', varName,'\n', arr[i,pindx,:,:,:])
    ds.close()
    return arr

def firstHorzAndVertInterpolation( nvars, npoints, xpos, ypos, zpos, xcoordVals, ycoordVals, zcoordVals, varValsAtCorners):
    varVals = np.zeros((nvars, npoints))
    for i in range(nvars):
        for j in range(npoints):
                ### remember bottom and top are opposite in GLORYS because z co-ordinates is depth co-ordinates
                # if i < 3 and j<1:
                #     # print(f'before interpolatoin var {varList[i]} \n', varValsAtCorners[i,j,0,:,:])
                #     # print(f'interpolation coords:')
                #     # print('xpos ypos', xpos[j], ypos[j])
                #     # print('xcorners', xcoordVals[j,:])
                #     # print('ycorners', ycoordVals[j,:])
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
    return varVals


def scipy3DLinearInterpolatoin( nvars, npoints, xpos, ypos, zpos, cartZvarIndxInVarList, xcoordVals, ycoordVals, zcoordVals, listVarValsAtCorners):
    # listVarValsAtCorners of shape (nvars, npoints, 2, 2, 2) 2 in vertical dir, 2 in south-north, 2 in west-east
    cartZ = listVarValsAtCorners[cartZvarIndxInVarList,:,:,:,:]
    varVals = np.zeros((nvars, npoints))

    for pIndx in range(npoints):
        ## unravel to 8 points
        xcornerPos = np.array([xcoordVals[pIndx,0],  #1
                            xcoordVals[pIndx,1],  #2
                            xcoordVals[pIndx,1],  #3
                            xcoordVals[pIndx,0],  #4
                            xcoordVals[pIndx,0],  #5
                            xcoordVals[pIndx,1],  #6
                            xcoordVals[pIndx,1],  #7
                            xcoordVals[pIndx,0]]) #8 ## anticlockWise in XY plane
        
        ycornerPos = np.array([ycoordVals[pIndx,0],  #1
                            ycoordVals[pIndx,0],  #2
                            ycoordVals[pIndx,1],  #3
                            ycoordVals[pIndx,1],  #4
                            ycoordVals[pIndx,0],  #5
                            ycoordVals[pIndx,0],  #6
                            ycoordVals[pIndx,1],  #7
                            ycoordVals[pIndx,1]]) #8 ## anticlockWise in XY plane
        
        sigmaCornerPos = np.array([zcoordVals[pIndx,0],  #1
                            zcoordVals[pIndx,0],  #2
                            zcoordVals[pIndx,0],  #3
                            zcoordVals[pIndx,0],  #4
                            zcoordVals[pIndx,1],  #5
                            zcoordVals[pIndx,1],  #6
                            zcoordVals[pIndx,1],  #7
                            zcoordVals[pIndx,1]]) #8  ## anticlockWise in XY plane and bottom to top
        
        zcornerPos = np.array([cartZ[pIndx,0,0,0],  #1
                            cartZ[pIndx,0,0,1],  #2
                            cartZ[pIndx,0,1,1],  #3
                            cartZ[pIndx,0,1,0],  #4
                            cartZ[pIndx,1,0,0],  #5
                            cartZ[pIndx,1,0,1],  #6
                            cartZ[pIndx,1,1,1],  #7
                            cartZ[pIndx,1,1,0]]) #8 ## anticlockWise in XY plane and bottom to top
        
        #first calculate z particle position
        interp = LinearNDInterpolator(list(zip(xcornerPos, ycornerPos, sigmaCornerPos)), zcornerPos)
        Z = interp(xpos[pIndx], ypos[pIndx], zpos[pIndx])
        # if pIndx == 0:
        #     print('zcornerPos',zcornerPos)
        #     print('xpos[pIndx], ypos[pIndx], zpos[pIndx]', xpos[pIndx], ypos[pIndx], zpos[pIndx])
        #     print('Z=',Z)
        
        for i in range(nvars):
            varValsAtCorners = np.array([listVarValsAtCorners[i,pIndx,0,0,0],  #1
                                listVarValsAtCorners[i,pIndx,0,0,1],  #2
                                listVarValsAtCorners[i,pIndx,0,1,1],  #3
                                listVarValsAtCorners[i,pIndx,0,1,0],  #4
                                listVarValsAtCorners[i,pIndx,1,0,0],  #5
                                listVarValsAtCorners[i,pIndx,1,0,1],  #6
                                listVarValsAtCorners[i,pIndx,1,1,1],  #7
                                listVarValsAtCorners[i,pIndx,1,1,0]]) #8 ## anticlockWise in XY plane and bottom to top
            
            interp = LinearNDInterpolator(list(zip(xcornerPos, ycornerPos, zcornerPos)), varValsAtCorners)
            varVals[i,pIndx] = interp(xpos[pIndx], ypos[pIndx], Z)

    return varVals


def getVarsAtPos(xpos, ypos, zpos, xcoord, ycoord, zcoord, cartZvarIndx, file, varList):
    #returns variable array of shape (nvars, npoints)
    # xpos, ypos and zpos are of shape (npoints)

    npoints = len(xpos)
    nvars = len(varList)

    Xindices, Yindices, Zindices = getNearestIndices(xpos, ypos, zpos, xcoord, ycoord, zcoord)
    # xcoord, ycoord, zcoord are 1D coordiante values in x, y and z 
    #Xindices, Yindices, Zindices all of shape (npoints,2)
    # print('xpos, ypos, zpos',xpos[0], ypos[0], zpos[0])
    # print('Xindices, Yindices, Zindices', Xindices[0], Yindices[0], Zindices[0])
    
    
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

    varVals = firstHorzAndVertInterpolation( nvars, npoints, xpos, ypos, zpos, xcoordVals, ycoordVals, zcoordVals, varValsAtCorners)
    #varVals = scipy3DLinearInterpolatoin( nvars, npoints, xpos, ypos, zpos, cartZvarIndx, xcoordVals, ycoordVals, zcoordVals, varValsAtCorners)
    # returns varVals of shape (nvars, npoints))

    

    varVals[:,maskParticle] = np.nan
    return varVals
            

def updatePositions(dt, xpos, ypos, zpos, xvel, yvel, zvel, dxi_z, deta_z, dsigma_z ):
    earthRad = 6.371e6
    lats = np.deg2rad(ypos)
    R = earthRad * np.cos(lats)

    dlon = np.rad2deg((xvel * dt.seconds)/R)
    dlat = np.rad2deg((yvel * dt.seconds)/earthRad)
    dz = zvel * dt.seconds
    # slope = np.sqrt(dxi_z**2 + deta_z**2)
    # horVel = np.sqrt(xvel**2 + yvel**2)
    # new_zvel = -(horVel*slope)/dsigma_z + zvel
    dsigma = dz/dsigma_z
    # print('In update Positions')
    # print(f'xvel, yvel, zvel = {xvel[0]}, {yvel[0]}, {zvel[0]}')
    print(f'dlon, dlat, dsigma = {dlon[0]}, {dlat[0]}, {dsigma[0]}')
    # print(f'xpos, ypos, zpos = {xpos[0]}, {ypos[0]}, {zpos[0]}')
    # print('\n')
    xpos += dlon
    ypos += dlat
    zpos += dsigma

    return xpos, ypos, zpos




        