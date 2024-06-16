import numpy as np
import argparse
from netCDF4 import Dataset
import sys
from pathlib import Path

def getIndices(cur_xpos, cur_ypos, xh, yh):
    ### gives out list of indices for x and y positions at nearlest lower left corner
    xlen = len(xh)
    ylen = len(yh)
    plen = len(cur_xpos)

    cur_xpos = np.around(cur_xpos, decimals=8)
    cur_ypos = np.around(cur_ypos, decimals=8)

    dx = xh[1] - xh[0]
    dy = yh[1] - yh[0]

    if len(cur_ypos)!= plen:
        print('xpos and ypos do not have same length')
        sys.exit()
    
    index_X = np.zeros((plen,), dtype = int)
    index_Y = np.zeros((plen,), dtype = int)
    
    index_X[:] = -999
    index_Y[:] = -999

    maskNan = np.isnan(cur_xpos[:]) + np.isnan(cur_ypos[:])

    outDomain = (cur_ypos[:] > yh[ylen-1]) + (cur_ypos[:] < yh[0])

    ## to avoid error while dividing convert the masked positions to be zero
    cur_xpos[maskNan] = 0.0
    cur_ypos[maskNan] = 0.0

    cur_xpos[outDomain] = 0.0
    cur_ypos[outDomain] = 0.0

    index_X = np.floor((cur_xpos[:] - xh[0])//dx)
    index_Y = np.floor((cur_ypos[:] - yh[0])//dy)

    index_X = index_X.astype(int)
    index_Y = index_Y.astype(int)

    ## making the masked values to -999

    index_X[maskNan] = -999
    index_Y[maskNan] = -999

    index_X[outDomain] = -999
    index_Y[outDomain] = -999
    
    return index_X, index_Y

def updateCyclicXdirection(cur_xpos, xh):
    ## sets the values for x to fall inside domain using periodicity in x-direction
    ## cur_xpos is array of x positions

    dx = xh[1] - xh[0]
    xlen = len(xh)
    toleranceVal = 1e-7

    ## to avoid errors for nan values
    maskNan = np.isnan(cur_xpos[:])
    cur_xpos[maskNan] = 0.0

    ## left and right masks for periodicity in x direction
    rightToleranceMask = np.abs((dx + xh[xlen-1]) - cur_xpos) < toleranceVal
    cur_xpos[rightToleranceMask] = xh[0]
    maskRight = cur_xpos >= (xh[xlen-1] + dx)
    cur_xpos[maskRight] = (cur_xpos[maskRight] - xh[xlen-1] - dx) + xh[0]

    leftToleranceMask = np.abs((xh[0] - dx) - cur_xpos) < toleranceVal
    cur_xpos[leftToleranceMask] = xh[xlen - 1]
    maskLeft = cur_xpos < xh[0]
    cur_xpos[maskLeft] = (xh[xlen -1] + dx) - (xh[0] - cur_xpos[maskLeft])

    # reassigning the nan values 
    cur_xpos[maskNan] = float('nan')

    return cur_xpos

def biLinearInterpolate(x, y, xh , yh, nArray, arrayStack):
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

    plen = len(x) ## number of particles
    xlen = len(xh)
    ylen = len(yh)

    dx = xh[1] - xh[0]
    dy = yh[1] - yh[0]


    x1, y1 = 0, 0
    x2, y2 = dx, 0
    x3, y3 = dx, dy
    x4, y4 = 0, dy

    x = updateCyclicXdirection(x, xh)
    x1_indx, y1_indx = getIndices(x, y, xh, yh)
    outDomain = x1_indx == -999
    # print('x indices \n',x1_indx)
    # print('y indices \n', y1_indx)
    # print('min indices', np.min(x1_indx), np.min(y1_indx))
    # print('max indices', np.max(x1_indx), np.max(y1_indx))
    #sys.exit()
    
    x2_indx = x1_indx[:] + 1
    y2_indx = y1_indx[:] + 1

    #periodic in x direction
    x2_indx[x2_indx == xlen] = 0

    outDomain = outDomain + (y2_indx == ylen)
    inDomain = ~outDomain

    x1_indx[outDomain] = 0
    x2_indx[outDomain] = 0
    y1_indx[outDomain] = 0
    y2_indx[outDomain] = 0

    try:
        x = x - xh[x1_indx]
        y = y - yh[y1_indx]
    except:
        mask = x1_indx > (xlen-1)
        print(x[mask])

    # shape function
    N1 = 1/(dx*dy) * (x - x2) * (y - y4)
    N2 = -1/(dx * dy) * (x - x1) * (y - y3)
    N3 = 1/(dx * dy) * (x - x4) * (y - y2)
    N4 = -1/(dx * dy) * (x - x3) * (y - y1)


    Q1 = arrayStack[:, y1_indx, x1_indx]
    Q2 = arrayStack[:, y1_indx, x2_indx]
    Q3 = arrayStack[:, y2_indx, x2_indx]
    Q4 = arrayStack[:, y2_indx, x1_indx]

    returnArr = N1 * Q1 + N2 * Q2 + N3 * Q3 + N4 * Q4
    returnArr[:,outDomain] = float('nan')

    return returnArr

    # returnArr = np.zeros((nArray, plen), dtype=np.float64)

    # for fldIndx in range(nArray):
    #     Q1 = np.zeros((plen,),dtype = float)
    #     Q2 = np.zeros((plen,),dtype = float)
    #     Q3 = np.zeros((plen,),dtype = float)
    #     Q4 = np.zeros((plen,),dtype = float)

    #     Q1[inDomain] = arrayStack[fldIndx, y1_indx[inDomain], x1_indx[inDomain]]
    #     Q2[inDomain] = arrayStack[fldIndx, y1_indx[inDomain], x2_indx[inDomain]]
    #     Q3[inDomain] = arrayStack[fldIndx, y2_indx[inDomain], x2_indx[inDomain]]
    #     Q4[inDomain] = arrayStack[fldIndx, y2_indx[inDomain], x1_indx[inDomain]]

    #     returnArr[fldIndx, inDomain] = N1[inDomain] * Q1[inDomain] + \
    #                                    N2[inDomain] * Q2[inDomain] + \
    #                                    N3[inDomain] * Q3[inDomain] + \
    #                                    N4[inDomain] * Q4[inDomain]

    # returnArr[:, outDomain] = float('nan')

    # return returnArr

def get_l_i_theta(theta_p, arr_theta_j, i_index):
    # this function will be used in the function lagrange4OPolyIntp
    # look JHU turbulence dataset eq 12, this function gives the value of l_theta
    # see page 6 of http://turbulence.pha.jhu.edu/docs/Database-functions.pdf

    ## arr_theta_j is the array of four positions for any of x, y or z direction for all the particles
    ## arr_theta_j has shape (nParticles, 4)
    ## returns array of shape (nParticles)

    plen = len(theta_p) ### number of positions

    num = np.ones((plen,),dtype=np.float64)
    den = np.ones((plen,),dtype=np.float64)
    for j in range(4):
        if j != i_index:
            num[:] *= theta_p[:] - arr_theta_j[:, j]
            den[:] *= arr_theta_j[:, i_index] - arr_theta_j[:, j]

    mask = den == 0
    den[mask] = 1
    returnArray = num/den
    returnArray[mask] = 0.0
    return returnArray
    
def get_fields_at_loc(x_pos, y_pos, xh , yh, nArray, stackArr):
    # This function interpolates the array is stackArray in positions(xpos, ypos)

    # look JHU turbulence dataset eq 12, this function gives the value of f(x_prime)
    # see page 6 of http://turbulence.pha.jhu.edu/docs/Database-functions.pdf

    #x_pos, y_pos are the position where the arrays are to be interpolated

    plen = len(x_pos)  # number of particles
    xlen = len(xh)
    ylen = len(yh)

    # four x positions indices involved in quadratic interpolation for each particle
    # two nodes are in the left of the point where the field is to be interpolated and two to the right
    x_pos_4arr_indices = np.zeros((plen, 4), dtype=int)

    # four y positions indices involved in quadratic interpolation for each particle
    # two nodes are below the point where the field is to be interpolated and two above
    y_pos_4arr_indices = np.zeros((plen, 4), dtype=int)

    x_pos = updateCyclicXdirection(x_pos, xh)

    ## point right at left and below of the locations
    x_pos_4arr_indices[:,1], y_pos_4arr_indices[:,1] = getIndices(x_pos, y_pos, xh, yh)

    #print('max xpos index :{0:4d}\n max ypos index :{1:4d}'.format(np.max(x_pos_4arr_indices[:,1]), 
    #np.max(y_pos_4arr_indices[:,1])))

    ## the points that are out of domain and have nan in postions have index -999
    outDomain = (y_pos_4arr_indices[:, 1] == -999)

    x_pos_4arr_indices[:, 0] = x_pos_4arr_indices[:, 1] - 1 
    y_pos_4arr_indices[:, 0] = y_pos_4arr_indices[:, 1] - 1
    
    x_pos_4arr_indices[:, 2] = x_pos_4arr_indices[:, 1] + 1
    y_pos_4arr_indices[:, 2] = y_pos_4arr_indices[:, 1] + 1
    
    x_pos_4arr_indices[:, 3] = x_pos_4arr_indices[:, 1] + 2
    y_pos_4arr_indices[:, 3] = y_pos_4arr_indices[:, 1] + 2

    ## set cyclic index in x direction
    x_pos_4arr_indices[x_pos_4arr_indices[:, 0] < 0, 0] = xlen - 1
    x_pos_4arr_indices[x_pos_4arr_indices[:, 2] == xlen, 2] = 0
    x_pos_4arr_indices[x_pos_4arr_indices[:, 3] == xlen, 3] = 0
    x_pos_4arr_indices[x_pos_4arr_indices[:, 3] > xlen, 3] = 1

    ## if the neboring points are out of domain add to the mask
    outDomain = outDomain + (y_pos_4arr_indices[:, 0] < 0 )
    outDomain = outDomain + (y_pos_4arr_indices[:, 2] >= ylen)
    outDomain = outDomain + (y_pos_4arr_indices[:, 3] >= ylen)
    inDomain = ~outDomain


    ## to aviod errors set the masked indices to zero
    x_pos_4arr_indices[outDomain, 0] = 0 
    y_pos_4arr_indices[outDomain, 0] = 0

    x_pos_4arr_indices[outDomain, 1] = 0
    y_pos_4arr_indices[outDomain, 1] = 0

    x_pos_4arr_indices[outDomain, 2] = 0
    y_pos_4arr_indices[outDomain, 2] = 0
    
    x_pos_4arr_indices[outDomain, 3] = 0
    y_pos_4arr_indices[outDomain, 3] = 0

    ## set 4 x 4 array for each particle positions
    Arr = np.zeros((nArray, plen, 4, 4 ), dtype=np.float64)
    for arrIndx in range(nArray):
        workArray = stackArr[arrIndx,:,:]
        for p in range(plen):
            if inDomain[p]:
                try:
                    Arr[arrIndx, p, :, :] = workArray[np.ix_(
                        y_pos_4arr_indices[p, :], x_pos_4arr_indices[p, :])]
                except:
                    print('for particle', p)
                    print('x pos indices', x_pos_4arr_indices[p, :])
                    print('y pos indices', y_pos_4arr_indices[p, :])
                    sys.exit()

    # interpolate array
    interpolatedVal = np.zeros((nArray, plen), dtype=np.float64)
    for arrIndx in range(nArray):        
        for j in range(3):
            l_y_j = get_l_i_theta(y_pos[:], yh[y_pos_4arr_indices], j)
            for i in range(3):
                l_x_i = get_l_i_theta(x_pos, xh[x_pos_4arr_indices], i)
                interpolatedVal[arrIndx, :] += Arr[arrIndx, :, j, i] * l_x_i * l_y_j

    ## setting the interpolated values for all fields for particles out of domain to NaN
    interpolatedVal[:, outDomain] = float('nan')

    #the retunValue is in shape (nArray, nParticles)
    return interpolatedVal

def get_field_at_time(time_pos, nfields, fieldVals, dt):
    # this function interpolates using cubit hermite interpolation in time
    # see page 19 of http://turbulence.pha.jhu.edu/docs/Database-functions.pdf

    # time_pos is the relative time position from the nearest left point at which the field is to be interplated
    # fieldVals are the value of the field for four time steps required for interpolating
    # dt is the time difference between the four fieldVals given
    # fieldVals are of shape (4, nfields, nparticles) or (4, ylen, xlen) depending on the input type
    

    # relative time position from the nearest left point
    
    n = 1
    a = fieldVals[n,:,:]
    b = (fieldVals[n+1,:,:] - fieldVals[n-1,:,:])/(2*dt)
    c = (fieldVals[n+1,:,:] - 2*fieldVals[n,:,:] + fieldVals[n-1,:,:])/(2* dt**2)
    d = (-fieldVals[n-1,:,:] + 3*fieldVals[n,:,:] - 3 * fieldVals[n+1,:,:] + fieldVals[n+2,:,:])/(2*dt**3 )


    returnVals = a  + \
                 b * time_pos + \
                 c * time_pos**2 + \
                 d * time_pos**2 * (dt - time_pos)

    # the returnValues are of shape (nFields, nPositions) or (ylen, xlen) depending on the input type
    return returnVals

def updatePositions_predictorCorrector(cur_xpos, cur_ypos, 
                                       cur_xvel_inGrid, cur_yvel_inGrid,
                                       next_xvel_inGrid, next_yvel_inGrid, 
                                       dt, xh, yh, intp='L4'):

    # this updates the particle positions using predictor corrector algorithm.
    
    curVelInGrid = np.stack((cur_xvel_inGrid, cur_yvel_inGrid), axis = 0)
    nextVelInGrid = np.stack((next_xvel_inGrid, next_yvel_inGrid), axis=0)

    cur_xpos = updateCyclicXdirection(cur_xpos, xh)
    if intp == 'L4':
        #print('L4 interpolation')
        curXvelYvel = get_fields_at_loc(cur_xpos, cur_ypos, xh, yh, 2, curVelInGrid)
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        curXvelYvel = biLinearInterpolate(cur_xpos, cur_ypos, xh, yh, 2, curVelInGrid)

    cur_xvel = curXvelYvel[0,:]/1000  # changing to km/sec
    cur_yvel = curXvelYvel[1,:]/1000  # changing to km/sec

    ##making nan values to zero to avoid error
    cur_nanMask = np.isnan(cur_xvel)

    cur_xpos[cur_nanMask] = 0.0
    cur_xvel[cur_nanMask] = 0.0

    cur_ypos[cur_nanMask] = 0.0
    cur_yvel[cur_nanMask] = 0.0
    
    predicted_xpos = cur_xpos + dt * cur_xvel
    predicted_ypos = cur_ypos + dt * cur_yvel

    if intp == 'L4':
        #print('L4 interpolation')
        nextXvelYvel = get_fields_at_loc(predicted_xpos, predicted_ypos, xh, yh, 2, nextVelInGrid)
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        nextXvelYvel = biLinearInterpolate(predicted_xpos, predicted_ypos, xh, yh, 2, nextVelInGrid)

    
    next_xvel = nextXvelYvel[0,:]/1000  # changing to km/sec
    next_yvel = nextXvelYvel[1,:]/1000  # changing to km/sec

    # masking nan values to zero avoid error
    next_nanMask = np.isnan(next_xvel)

    next_xvel[next_nanMask] = 0.0
    next_yvel[next_nanMask] = 0.0

    next_xpos = cur_xpos + 0.5* dt * (cur_xvel + next_xvel)
    next_ypos = cur_ypos + 0.5* dt * (cur_yvel + next_yvel)

    # print('max x position moved', np.max(next_xpos - cur_xpos))
    # print('max y position moved', np.max(next_ypos - cur_ypos))
    #reinstating the nan values

    nanMask = cur_nanMask + next_nanMask
    next_xpos[nanMask] = float('nan')
    next_ypos[nanMask] = float('nan')
    
    return next_xpos, next_ypos

def getInterPolatedFieldsAtLocAndTime(x_pos, y_pos, time_pos, dt, xh, yh, nFields, fieldVals):
    # This function provides with the both time and positions
    # Time interpolation is done after space interpolation
    # nFields is of shape (nFields, 4, ylen, xlen) and this field is in grid
    # tpos is relative time position from t2. There is always two time values before tpos and two timevalues after tpos

    fields_at_t1 = get_fields_at_loc(x_pos, y_pos, xh, yh, nFields, fieldVals[:,0,:,:])
    fields_at_t2 = get_fields_at_loc(x_pos, y_pos, xh, yh, nFields, fieldVals[:,1,:,:])
    fields_at_t3 = get_fields_at_loc(x_pos, y_pos, xh, yh, nFields, fieldVals[:,2,:,:])
    fields_at_t4 = get_fields_at_loc(x_pos, y_pos, xh, yh, nFields, fieldVals[:,3,:,:])

    #fieldsStack is of shape(4, nFields, nParticles)
    fieldStacks = np.stack((fields_at_t1, fields_at_t2, fields_at_t3, fields_at_t4), axis=0)

    #interPolatedFields is of shape(nFields, nParticles)
    interPolatedFields = get_field_at_time(time_pos, nFields, fieldStacks, dt)

    return interPolatedFields

def updatePositions_RK4(cur_xpos, cur_ypos,
                        xvel_inGrid, yvel_inGrid,
                        dt, xh, yh, intp='L4'):

    # this function updates the position using RK-4 method

    # this function requires fieldVals for 3 time locations at t_n , t_(n+1/2) and t_(n + 1)
    # the shape of xvel_inGrid and yvel_inGrid is (3, ylen, xlen)

    stackedFields = np.stack((xvel_inGrid, yvel_inGrid), axis = 0)

    cur_xpos = updateCyclicXdirection(cur_xpos, xh)
    if intp == 'L4':
        #print('L4 interpolation')
        curXvelYvel = get_fields_at_loc(cur_xpos, cur_ypos, xh, yh, 2, stackedFields[:,0,:,:])
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        curXvelYvel = biLinearInterpolate(cur_xpos, cur_ypos, xh, yh, 2, stackedFields[:,0,:,:])
    
    
    cur_xvel = curXvelYvel[0, :]
    cur_yvel = curXvelYvel[1, :]

    ##making nan values to zero to avoid error
    cur_nanMask = np.isnan(cur_xvel)

    cur_xpos[cur_nanMask] = 0.0
    cur_xvel[cur_nanMask] = 0.0

    cur_ypos[cur_nanMask] = 0.0
    cur_yvel[cur_nanMask] = 0.0

    k1 = cur_xvel/1000 #changing to km/sec
    l1 = cur_yvel/1000 #changing to km/sec

    new_xpos = cur_xpos + 0.5*dt * k1
    new_ypos = cur_ypos + 0.5*dt * l1
    
    ## getting velocity at half time step at new xpos, ypos
    if intp == 'L4':
        #print('L4 interpolation')
        curXvelYvel = get_fields_at_loc(new_xpos, new_ypos, xh, yh, 2, stackedFields[:,1,:,:])
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        curXvelYvel = biLinearInterpolate(new_xpos, new_ypos, xh, yh, 2, stackedFields[:, 1, :, :])

    cur_xvel = curXvelYvel[0, :]
    cur_yvel = curXvelYvel[1, :]

    ##making nan values to zero to avoid error
    cur_nanMask = cur_nanMask + np.isnan(cur_xvel)

    cur_xpos[cur_nanMask] = 0.0
    cur_xvel[cur_nanMask] = 0.0

    cur_ypos[cur_nanMask] = 0.0
    cur_yvel[cur_nanMask] = 0.0

    k2 = cur_xvel/1000 #changing to km/sec
    l2 = cur_yvel/1000 #changing to km/sec

    new_xpos = cur_xpos + 0.5*dt * k2
    new_ypos = cur_ypos + 0.5*dt * l2

    ## getting velocity at half time step at new xpos, ypos
    if intp == 'L4':
        #print('L4 interpolation')
        curXvelYvel = get_fields_at_loc(new_xpos, new_ypos, xh, yh, 2, stackedFields[:,1,:,:])
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        curXvelYvel = biLinearInterpolate(new_xpos, new_ypos, xh, yh, 2, stackedFields[:, 1, :, :])

    cur_xvel = curXvelYvel[0, :]
    cur_yvel = curXvelYvel[1, :]

    ##making nan values to zero to avoid error
    cur_nanMask = cur_nanMask + np.isnan(cur_xvel)

    cur_xpos[cur_nanMask] = 0.0
    cur_xvel[cur_nanMask] = 0.0

    cur_ypos[cur_nanMask] = 0.0
    cur_yvel[cur_nanMask] = 0.0

    k3 = cur_xvel/1000  # changing to km/sec
    l3 = cur_yvel/1000  # changing to km/sec

    new_xpos = cur_xpos + dt * k3
    new_ypos = cur_ypos + dt * l3

    ## getting velocity at new time step at new xpos, ypos
    if intp == 'L4':
        #print('L4 interpolation')
        curXvelYvel = get_fields_at_loc(new_xpos, new_ypos, xh, yh, 2, stackedFields[:,2,:,:])
    elif intp == 'bilinear':
        #print('bilinear interpolation')
        curXvelYvel = biLinearInterpolate(new_xpos, new_ypos, xh, yh, 2, stackedFields[:, 2, :, :])

    cur_xvel = curXvelYvel[0, :]
    cur_yvel = curXvelYvel[1, :]

    ##making nan values to zero to avoid error
    cur_nanMask = cur_nanMask + np.isnan(cur_xvel)

    cur_xpos[cur_nanMask] = 0.0
    cur_xvel[cur_nanMask] = 0.0

    cur_ypos[cur_nanMask] = 0.0
    cur_yvel[cur_nanMask] = 0.0

    k4 = cur_xvel/1000 #changing to km/sec
    l4 = cur_yvel/1000 #changing to km/sec

    next_xpos = cur_xpos + 1/6 * dt * (k1 + 2*k2 +2*k3 + k4)
    next_ypos = cur_ypos + 1/6 * dt * (l1 + 2*l2 +2*l3 + l4)

    # print('max x position moved', np.max(next_xpos - cur_xpos))
    # print('max y position moved', np.max(next_ypos - cur_ypos))

    next_xpos[cur_nanMask] = float('nan')
    next_ypos[cur_nanMask] = float('nan')

    return next_xpos, next_ypos
