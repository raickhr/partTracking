from netCDF4 import Dataset
import numpy as np
from scipy import signal, interpolate
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

def get_Kernel(filterLength, gridSizeX, gridSizeY):
    #All the inputs are to be given in KM
    
    kernelSizeX = filterLength // gridSizeX + 1
 
    kernelSizeY = filterLength // gridSizeY + 1

    print('kernel size for Filtering ', kernelSizeX, kernelSizeY)
    print('Filtering to ',filterLength, ' km, gridSizeIn Km =', gridSizeX, gridSizeY)

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
    
    gradx = (np.roll(field, -1, axis=xaxis) - np.roll(field, 1, axis=xaxis))/(2*dx)
    grady = (np.roll(field, -1, axis=yaxis) - np.roll(field, 1, axis=yaxis))/(2*dy)
    
    return gradx, grady


def filterInLocal(localEllList, localVarList, allVars, Xlen, Ylen, timeLen):
    nLocalEll = len(localEllList)
    nLocalVars = len(localVarList)
    filteredArry = np.zeros(
        (nLocalEll, nLocalVars, timeLen, Ylen, Xlen), dtype=float)

    for i in range(nLocalEll):
        ell = localEllList[i]
        for j in range(nLocalVars):
            var = localVarList[j]
            filteredArry[i, j, :, :, :] = get_filtered_Field(
                allVars[var,:,:,:], ell, dxH, dyH)

    return filteredArry

timeLen = 0
Xlen = 0
Ylen = 0
dxH = 0.0
dyH = 0.0

if rank == 0:
    fldLoc = '/scratch/srai6/MOM6/postProcessing/modelTruthComparison'
    fileName = 'prog_100_instants.nc'
    ds=Dataset(fldLoc + '/' + fileName)

    xqNC = ds.variables['xq']
    yqNC = ds.variables['yq']
    xhNC = ds.variables['xh']
    yhNC = ds.variables['yh']

    u = np.array(ds.variables['u'])[:, 0, :, :]#(xq, yh)
    v = np.array(ds.variables['v'])[:, 0, :, :]#(xh, yq)
    h = np.array(ds.variables['h'])[:, 0, :, :]#(xh, yh)

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

    timeVal = np.array(ds.variables['Time'])
    timeUnits = ds.variables['Time'].units

    Ylen, Xlen = len(yh), len(xh)
    timeLen = len(timeVal)

timeLen = comm.bcast(timeLen, root=0)
Xlen = comm.bcast(Xlen, root=0)
Ylen = comm.bcast(Ylen, root=0)

U = np.zeros((timeLen, Ylen, Xlen), dtype=float)
V = np.zeros((timeLen, Ylen, Xlen), dtype=float)
h = np.zeros((timeLen, Ylen, Xlen), dtype=float)
hU = np.zeros((timeLen, Ylen, Xlen), dtype=float)
hV = np.zeros((timeLen, Ylen, Xlen), dtype=float)
timeVal = np.zeros((timeLen), dtype=float)
xh = np.zeros((Xlen), dtype=float)
yh = np.zeros((Ylen), dtype=float)

U = comm.bcast(U, root=0)
V = comm.bcast(V, root=0)
h = comm.bcast(h, root=0)
hU = comm.bcast(hU, root=0)
hV = comm.bcast(hV, root=0)
dxH = comm.bcast(dxH, root=0)
dyH = comm.bcast(dyH, root=0)
timeVal = comm.bcast(timeVal, root=0)

yh = comm.bcast(yh, root=0)
xh = comm.bcast(xh, root=0)

allVars = np.stack((U,V, h, hU, hV), axis=0)
varList = np.arange(5)

ellList = np.arange(50,850,50)
ellLen = len(ellList)


localEllList = []
localVarList = []

if rank == 1:
    localEllList = np.arange(50, 350, 50, dtype='i')
    localVarList = np.arange(0,5, dtype='i')
elif rank == 2:
    localEllList = np.arange(350, 550, 50, dtype='i')
    localVarList = np.arange(0, 5, dtype='i')
elif rank == 3:
    localEllList = np.arange(550, 650, 50, dtype='i')
    localVarList = np.arange(0, 5, dtype='i')
elif rank == 4:
    localEllList = np.arange(650, 700, 50, dtype='i')
    localVarList = np.arange(0, 5, dtype='i')

elif rank >= 5 and rank < 10:
    localEllList = np.arange(700, 750, 50, dtype='i')
    localVarList = np.array([rank % 5], dtype= 'i')

elif rank >= 10 and rank < 15:
    localEllList = np.arange(750, 800, 50, dtype='i')
    localVarList = np.array([rank % 5], dtype = 'i')

elif rank >= 15 and rank < 20:
    localEllList = np.arange(800, 850, 50, dtype='i')
    localVarList = np.array([rank % 5], dtype = 'i')

nLocalEll = 0
nLocalVars = 0

if rank != 0:
    filteredArry = filterInLocal(localEllList, localVarList, allVars, Xlen, Ylen, timeLen)
    
    comm.send(nLocalVars, dest=0, tag=77)
    comm.send(nLocalEll, dest=0, tag=78)
    
    comm.Send([localVarList, MPI.INT], dest=0, tag=87)
    comm.Send([localEllList, MPI.INT], dest=0, tag=88)
    
    nLocalEll = len(localEllList)
    nLocalVars = len(localVarList)

    comm.Send(filteredArry, dest=0, tag=10*rank)
    

else:
    allUbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
    allVbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
    allhbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
    allhUbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)
    allhVbar = np.zeros((ellLen, timeLen, Ylen, Xlen), dtype=float)

    for source in range(1,nprocs):

        nLocalVars = comm.recv(source=source, tag=77)
        nLocalEll  = comm.recv(source=source, tag=78)
        
        localVarList = np.arange(nLocalVars, dtype='i')
        localEllList = np.arange(nLocalEll, dtype='i')

        comm.Recv([localVarList, MPI.INT], source=source, tag=87)
        comm.Recv([localEllList, MPI.INT], source=source, tag=88)
        
        filteredArry = np.zeros((nLocalEll, nLocalVars, timeLen, Ylen, Xlen), dtype=np.float64)
        comm.Recv(filteredArry, source=source, tag=10*source)

        for i in range(nLocalEll):
            ell = localEllList[i]
            ellIndx = np.where(ellList == ell)[0][0]
            for j in range(nLocalVars):
                var = localVarList[j]    
                if var == 0:
                    allUbar[ellIndx, :, :, :] = filteredArry[i,j,:,:,:]

                elif var == 1:
                    allVbar[ellIndx, :, :, :] = filteredArry[i, j, :, :, :]

                elif var == 2:
                    allhbar[ellIndx, :, :, :] = filteredArry[i, j, :, :, :]

                elif var == 3:
                    allhUbar[ellIndx, :, :, :] = filteredArry[i, j, :, :, :]

                elif var == 4:
                    allhVbar[ellIndx, :, :, :] = filteredArry[i, j, :, :, :]


    writeFileName = '/scratch/srai6/MOM6/postProcessing/modelTruthComparison/filteredFieldsForPhilipsTauAndModel.nc'
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

    wcdf_Ubar = writeDS.createVariable('U', np.float32, ('ell', 'Time', 'yh', 'xh'))
    wcdf_Ubar.long_name = "Zonal Velocity"
    wcdf_Ubar.units = "ms-1"
    wcdf_Ubar[:, :, :, :] = allUbar[:, :, :, :]

    wcdf_Vbar = writeDS.createVariable('V', np.float32, ('ell', 'Time', 'yh', 'xh'))
    wcdf_Vbar.long_name = "Meridional Velocity"
    wcdf_Vbar.units = "ms-1"
    wcdf_Vbar[:, :, :, :] = allVbar[:, :, :, :]

    wcdf_hbar = writeDS.createVariable('h', np.float32, ('ell', 'Time', 'yh', 'xh'))
    wcdf_hbar.long_name = "Top Layer Height"
    wcdf_hbar.units = "m"
    wcdf_hbar[:, :, :, :] = allhbar[:, :, :, :]

    wcdf_hUbar = writeDS.createVariable('hU_bar', np.float32, ('ell', 'Time', 'yh', 'xh'))
    wcdf_hUbar.long_name = "product of h and U filtered"
    wcdf_hUbar.units = "m^2/sec"
    wcdf_hUbar[:, :, :, :] = allhUbar[:, :, :, :]

    wcdf_hVbar = writeDS.createVariable('hV_bar', np.float32, ('ell', 'Time', 'yh', 'xh'))
    wcdf_hVbar.long_name = "product of h and V filtered"
    wcdf_hVbar.units = "m^2/sec"
    wcdf_hVbar[:, :, :, :] = allhVbar[:, :, :, :]

    writeDS.close()

