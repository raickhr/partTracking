from filteringFunctions import *
import numpy as np
from netCDF4 import Dataset
import argparse
import sys

dxInKm = 5
dyInKm = 5

parser = argparse.ArgumentParser()

parser.add_argument("--inputFile", "-f", type=str, default='prog_RequiredFieldsOnly.nc', action='store',
                    help="this is the output file from MOM6")

parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")

parser.add_argument("--ellInKm", "-e", type=int, default=100, action='store',
                    help="this is the filterlength")

args = parser.parse_args()

fileName = args.inputFile
fldLoc = args.fldLoc
ellInKm = args.ellInKm

readFileName = fldLoc + '/' + fileName
writeFileName = fldLoc + '/' + \
    fileName.replace('_RequiredFieldsOnly_4O.nc', '_FilteredFields_0ell_4O.nc')

ds = Dataset(readFileName)

xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

U = np.array(ds.variables['u'][:, :, :], dtype=float)
V = np.array(ds.variables['v'][:, :, :], dtype=float)
h = np.array(ds.variables['h'][:, :, :], dtype=float)
# this is for first layer only
P = np.array(ds.variables['e'][:, :, :], dtype=float) * 9.81 * 1031.0

timeLen, Ylen, Xlen = np.shape(U)
print('shape of the array U ', timeLen, Ylen, Xlen)
sys.stdout.flush()

farr = np.ones((Ylen, Xlen), dtype=float)
f0 = 6.49e-05
beta = 2.0E-11

y = yh - np.mean(yh)

for i in range(Ylen):
    farr[i, :] = f0 + beta*y[i]


## f*U
farr = np.repeat(farr[np.newaxis, :, :], timeLen, axis=0)
fU = U * farr
fV = V * farr

#Velocity gradients calculations
dx_U, dy_U = getGradient(U, dxInKm*1000, dyInKm*1000)
dx_V, dy_V = getGradient(V, dxInKm*1000, dyInKm*1000)

dx_h, dy_h = getGradient(h, dxInKm*1000, dyInKm*1000)

U_bar = U
V_bar = V
hfU_bar = h * fU
hfV_bar = h * fV
UU_bar = U * U
VV_bar = V * V
UV_bar = U * V
hU_bar = h * U
hV_bar = h * V
hUU_bar = h * U * U
hUV_bar = h * U * V
hVV_bar = h * V * V
h_bar = h
P_bar = P
hP_bar = h * P
Pdx_h_bar = P * dx_h
Pdy_h_bar = P * dy_h

sys.stdout.flush()

writeDS = Dataset(writeFileName, 'w', format='NETCDF4_CLASSIC')

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

wcdf_U_bar = writeDS.createVariable(
    'u_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_U_bar.long_name = "u_bar"
wcdf_U_bar.units = "m s^-1"
wcdf_U_bar[:, :, :] = U_bar[:, :, :]

wcdf_V_bar = writeDS.createVariable(
    'v_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_V_bar.long_name = "v_bar"
wcdf_V_bar.units = "m s^-1"
wcdf_V_bar[:, :, :] = V_bar[:, :, :]

wcdf_hfU_bar = writeDS.createVariable(
    'hfu_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hfU_bar.long_name = "hfu_bar"
wcdf_hfU_bar.units = "m^2 s^-2"
wcdf_hfU_bar[:, :, :] = hfU_bar[:, :, :]

wcdf_hfV_bar = writeDS.createVariable(
    'hfv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hfV_bar.long_name = "hfv_bar"
wcdf_hfV_bar.units = "m^2 s^-2"
wcdf_hfV_bar[:, :, :] = hfV_bar[:, :, :]

wcdf_h_bar = writeDS.createVariable(
    'h_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_h_bar.long_name = "h_bar"
wcdf_h_bar.units = "m"
wcdf_h_bar[:, :, :] = h_bar[:, :, :]

wcdf_hU_bar = writeDS.createVariable(
    'hu_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hU_bar.long_name = "hu_bar"
wcdf_hU_bar.units = "m^2 s^-1"
wcdf_hU_bar[:, :, :] = hU_bar[:, :, :]

wcdf_hV_bar = writeDS.createVariable(
    'hv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hV_bar.long_name = "hv_bar"
wcdf_hV_bar.units = "m^2 s^-1"
wcdf_hV_bar[:, :, :] = hV_bar[:, :, :]

wcdf_hUU_bar = writeDS.createVariable(
    'huu_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hUU_bar.long_name = "huu_bar"
wcdf_hUU_bar.units = "m^3 s^-2"
wcdf_hUU_bar[:, :, :] = hUU_bar[:, :, :]

wcdf_hVV_bar = writeDS.createVariable(
    'hvv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hVV_bar.long_name = "hvv_bar"
wcdf_hVV_bar.units = "m^3 s^-2"
wcdf_hVV_bar[:, :, :] = hVV_bar[:, :, :]

wcdf_hUV_bar = writeDS.createVariable(
    'huv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hUV_bar.long_name = "huv_bar"
wcdf_hUV_bar.units = "m^3 s^-2"
wcdf_hUV_bar[:, :, :] = hUV_bar[:, :, :]

wcdf_UU_bar = writeDS.createVariable(
    'uu_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_UU_bar.long_name = "uu_bar"
wcdf_UU_bar.units = "m^2 s^-2"
wcdf_UU_bar[:, :, :] = UU_bar[:, :, :]

wcdf_VV_bar = writeDS.createVariable(
    'vv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_VV_bar.long_name = "vv_bar"
wcdf_VV_bar.units = "m^2 s^-2"
wcdf_VV_bar[:, :, :] = VV_bar[:, :, :]

wcdf_UV_bar = writeDS.createVariable(
    'uv_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_UV_bar.long_name = "uv_bar"
wcdf_UV_bar.units = "m^2 s^-2"
wcdf_UV_bar[:, :, :] = UV_bar[:, :, :]

wcdf_P_bar = writeDS.createVariable(
    'p_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_P_bar.long_name = "p_bar"
wcdf_P_bar.units = "m^2 s^-2"
wcdf_P_bar[:, :, :] = P_bar[:, :, :]

wcdf_Pdx_h_bar = writeDS.createVariable(
    'pdx_h_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_Pdx_h_bar.long_name = "pdx_bar"
wcdf_Pdx_h_bar.units = "m^2 s^-2"
wcdf_Pdx_h_bar[:, :, :] = Pdx_h_bar[:, :, :]

wcdf_Pdy_h_bar = writeDS.createVariable(
    'pdy_h_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_Pdy_h_bar.long_name = "pdy_bar"
wcdf_Pdy_h_bar.units = "m^2 s^-2"
wcdf_Pdy_h_bar[:, :, :] = Pdy_h_bar[:, :, :]

wcdf_hP_bar = writeDS.createVariable(
    'hp_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_hP_bar.long_name = "hp_bar"
wcdf_hP_bar.units = "m^3 s^-2"
wcdf_hP_bar[:, :, :] = hP_bar[:, :, :]

writeDS.close()

sys.exit()
