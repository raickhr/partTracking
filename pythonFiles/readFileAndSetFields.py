from netCDF4 import Dataset
import numpy as np
from scipy import signal, interpolate
from filteringFunctions import *
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--inputFile", "-f", type=str, default='prog.nc', action='store',
                    help="this is the output file from MOM6")

parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")


args = parser.parse_args()

fileName = args.inputFile
fldLoc = args.fldLoc

fileNum = int(fileName.lstrip('prog_').rstrip('.nc'))
nextFileNum = fileNum + 1
nextFile = fldLoc + '/prog_{0:03d}.nc'.format(nextFileNum)
nextFilePath = Path(nextFile)

ds = Dataset(fldLoc + '/' + fileName)

writeFileName = fldLoc + '/' + fileName.rstrip('.nc') +'_RequiredFieldsOnly_4O.nc'


xqNC = ds.variables['xq']   # q grid point x direction
yqNC = ds.variables['yq']   # q grid point y direction
xhNC = ds.variables['xh']   # h grid point x direction 
yhNC = ds.variables['yh']   # h grid point y direction

# v is between two q points in same horizontal line (x - axis)
# u is between two q points in same vertical line (y - axis)

u = np.array(ds.variables['u'])[:, 0, :, :]  # (xq, yh) grid
v = np.array(ds.variables['v'])[:, 0, :, :]  # (xh, yq) grid
h = np.array(ds.variables['h'])[:, 0, :, :]  # (xh, yh) grid
e = np.array(ds.variables['e'])[:, 0, :, :] # (xh, yh) grid
rv = np.array(ds.variables['RV'])[:, 0, :, :] # (yq, xq) grid
pv = np.array(ds.variables['PV'])[:, 0, :, :]  # (yq, xq) grid

u = fillInvalidWithZero(u)
v = fillInvalidWithZero(v)
h = fillInvalidWithZero(h)
e = fillInvalidWithZero(e)
rv = fillInvalidWithZero(rv)
pv = fillInvalidWithZero(pv)

timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units
timeLen = len(timeVal)

timeStep = (timeVal[1] - timeVal[0]) ##* 24 * 3600  ## seconds
print('time Step', timeStep)

xq = np.array(xqNC)
yq = np.array(yqNC)

xh = np.array(xhNC)
yh = np.array(yhNC)

dxH = (xh[1] - xh[0]) * 1000
dyH = (yh[1] - yh[0]) * 1000

dxQ = (xq[1] - xq[0]) * 1000
dyQ = (yq[1] - yq[0]) * 1000

print(dxH, dyH, dxQ, dyQ)

U = getInterpolated(u, xq, yh, xh, yh)  # Interpolating to same co-ordinates as h
V = getInterpolated(v, xh, yq, xh, yh)
RV = getInterpolated(rv, xq, yq, xh, yh)
PV = getInterpolated(pv, xq, yq, xh, yh)

print(np.shape(yh), np.shape(xh), np.shape(u))

writeDS = Dataset(writeFileName, 'w', format = 'NETCDF4_CLASSIC')

writeDS.createDimension('Time', None)
writeDS.createDimension('xh', 240)
writeDS.createDimension('yh', 320)

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

wcdf_U = writeDS.createVariable('u', np.float32, ('Time', 'yh', 'xh'))
wcdf_U.long_name = "Surface Zonal velocity"
wcdf_U.units = "m s-1"
wcdf_U[:,:,:] = U[:,:,:]

wcdf_V = writeDS.createVariable('v', np.float32, ('Time', 'yh', 'xh'))
wcdf_V.long_name = "Surface Meridional velocity"
wcdf_V.units = "m s-1"
wcdf_V[:,:,:] = V[:,:,:]

wcdf_e = writeDS.createVariable('e', np.float32, ('Time', 'yh', 'xh'))
wcdf_e.long_name = "Surface Interface Height Relative to Mean Sea Level"
wcdf_e.units = "m"
wcdf_e[:,:,:] = e[:,:,:]

wcdf_h = writeDS.createVariable('h', np.float32, ('Time', 'yh', 'xh'))
wcdf_h.long_name = "Surface Layer Thickness"
wcdf_h.units = "m"
wcdf_h[:, :, :] = h[:, :, :]

wcdf_RV = writeDS.createVariable('RV', np.float32, ('Time', 'yh', 'xh'))
wcdf_RV.long_name = "Relative Vorticity"
wcdf_RV.units = "s-1"
wcdf_RV[:, :, :] = RV[:, :, :]

wcdf_PV = writeDS.createVariable('PV', np.float32, ('Time', 'yh', 'xh'))
wcdf_PV.long_name = "Potential Vorticity"
wcdf_PV.units = "s-1"
wcdf_PV[:, :, :] = PV[:, :, :]

writeDS.close()





