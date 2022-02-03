from filteringFunctions import *
from netCDF4 import Dataset
import os.path
from os import path
import argparse

dxInKm = 5
dyInKm = 5

parser = argparse.ArgumentParser()

parser.add_argument("--inputFile", "-f", type=str, default='prog_RequiredFieldsOnly_4FilteredFields.nc', action='store',
                    help="this is the output file after filtering")

parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")

parser.add_argument("--ellInKm", "-e", type=int, default=100, action='store',
                    help="this is the filterlength")

parser.add_argument("--rho", "-r", type=float, default=1031.0, action='store',
                    help="this is the density of the layer")

args = parser.parse_args()

fileName = args.inputFile
fldLoc = args.fldLoc
ellInKm = args.ellInKm
rho = args.rho

readFileName = fldLoc + '/' + fileName
readSuffix = '_FilteredFields_{0:03d}km_4O.nc'.format(ellInKm)
fnum = ""
for s in fileName[0:8]:
    if s.isdigit():
        fnum += s
fnum = int(fnum)
nextReadFileName = fldLoc + '/prog_{0:03d}'.format(fnum + 1) + readSuffix


writeSuffix = '_EnergyEqnTerms_{0:03d}km_4O.nc'.format(ellInKm)
writeFileName = fldLoc + '/' + \
    fileName.replace(readSuffix, writeSuffix)

ds = Dataset(readFileName)


xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

dt = timeVal[1] - timeVal[0]
tlen = len(timeVal)

hU_bar = np.array(ds.variables['hu_bar'])
hV_bar = np.array(ds.variables['hv_bar'])
h_bar = np.array(ds.variables['h_bar'])

U_tilde = hU_bar/h_bar
V_tilde = hV_bar/h_bar

divU_tilde = getDiv(U_tilde, V_tilde, dxInKm*1000, dyInKm*1000)

KE_tilde = 0.5 * (U_tilde**2 + V_tilde**2)
d_dt_KE_tilde = (np.roll(KE_tilde, -1, axis=0) - KE_tilde)/dt


d_dx_KE_tilde, d_dy_KE_tilde = getGradient(KE_tilde, dxInKm*1000, dyInKm*1000)
advecTermKE_tilde = U_tilde * d_dx_KE_tilde + V_tilde * d_dy_KE_tilde

d_dt_h_bar = (np.roll(h_bar, -1, axis=0) - h_bar)/dt
d_dx_h_bar, d_dy_h_bar = getGradient(h_bar, dxInKm*1000, dyInKm*1000)
advecTermh_bar = U_tilde * d_dx_h_bar + V_tilde * d_dy_h_bar




if path.exists(nextReadFileName):
    ds2 = Dataset(nextReadFileName)
    hU_bar_next = np.array(ds2.variables['hu_bar'][0:2, :, :])
    hV_bar_next = np.array(ds2.variables['hv_bar'][0:2, :, :])
    h_bar_next = np.array(ds2.variables['h_bar'][0:2, :, :])
    U_tilde_next = hU_bar_next/h_bar_next
    V_tilde_next = hV_bar_next/h_bar_next

    KE_tilde_next = 0.5 * (U_tilde_next**2 + V_tilde_next**2)
    d_dx_KE_tilde_next, d_dy_KE_tilde_next = getGradient(KE_tilde_next, dxInKm*1000, dyInKm*1000)
    d_dt_KE_tilde[tlen-1, :, :] = (KE_tilde_next[0, :, :]- KE_tilde[tlen-1, :, :])/dt
    
    d_dx_h_bar_next, d_dy_h_bar_next = getGradient(h_bar_next, dxInKm*1000, dyInKm*1000)
    d_dt_h_bar[tlen-1, :, :] = (h_bar_next[0, :, :]- h_bar[tlen-1, :, :])/dt
    
    ds2.close()

else:
    print('filling the last time derivative with nan value')
    d_dt_KE_tilde[tlen-1, :, :] = float('nan')
    d_dt_h_bar[tlen-1, :, :] = float('nan')



writeDS = Dataset(writeFileName, 'w', format='NETCDF4_CLASSIC')

writeDS.createDimension('Time', None)
writeDS.createDimension('xh', 240)
writeDS.createDimension('yh', 320)

wcdf_Xh = writeDS.createVariable('xh', np.float64, ('xh'))
wcdf_Xh.long_name = 'h point nominal longitude'
wcdf_Xh.units = 'kilometers'
wcdf_Xh[:] = xh[:]

wcdf_Yh = writeDS.createVariable('yh', np.float64, ('yh'))
wcdf_Yh.long_name = 'h point nominal latitude'
wcdf_Yh.units = 'kilometers'
wcdf_Yh[:] = yh[:]

wcdf_Time = writeDS.createVariable('Time', np.float64, ('Time'))
wcdf_Time.long_name = "Time"
wcdf_Time.units = timeUnits
wcdf_Time.cartesian_axis = "T"
wcdf_Time.calendar_type = "JULIAN"
wcdf_Time.calendar = "JULIAN"
wcdf_Time[:] = timeVal

wcdf_U_tilde = writeDS.createVariable(
    'u_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_U_tilde.long_name = "u_tilde"
wcdf_U_tilde.units = "m s^-1"
wcdf_U_tilde[:, :, :] = U_tilde[:, :, :]

wcdf_V_tilde = writeDS.createVariable(
    'v_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_V_tilde.long_name = "v_tilde"
wcdf_V_tilde.units = "m s^-1"
wcdf_V_tilde[:, :, :] = V_tilde[:, :, :]

wcdf_KE_tilde = writeDS.createVariable(
    'KE_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_KE_tilde.long_name = "KE_tilde"
wcdf_KE_tilde.units = "m^2 s^-2"
wcdf_KE_tilde[:, :, :] = KE_tilde[:, :, :]

wcdf_d_dt_KE_tilde = writeDS.createVariable(
    'd_dt_KE_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_KE_tilde.long_name = "d_dt KE_tilde"
wcdf_d_dt_KE_tilde.units = "m^2 s^-3"
wcdf_d_dt_KE_tilde[:, :, :] = d_dt_KE_tilde[:, :, :]

wcdf_advecTermKE_tilde = writeDS.createVariable(
    'advecTermKE_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermKE_tilde.long_name = "advection KE_tilde"
wcdf_advecTermKE_tilde.units = "m^2 s^-3"
wcdf_advecTermKE_tilde[:, :, :] = advecTermKE_tilde[:, :, :]

wcdf_h_bar = writeDS.createVariable(
    'h_bar', np.float64, ('Time', 'yh', 'xh'))
wcdf_h_bar.long_name = "h_bar"
wcdf_h_bar.units = "m"
wcdf_h_bar[:, :, :] = h_bar[:, :, :]

wcdf_d_dt_h_bar = writeDS.createVariable(
    'd_dt_h_bar', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_h_bar.long_name = "d_dt h_bar"
wcdf_d_dt_h_bar.units = "m s^-1"
wcdf_d_dt_h_bar[:, :, :] = d_dt_h_bar[:, :, :]

wcdf_advecTermh_bar = writeDS.createVariable(
    'advecTermh_bar', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermh_bar.long_name = "advection h_bar"
wcdf_advecTermh_bar.units = "m s^-1"
wcdf_advecTermh_bar[:, :, :] = advecTermh_bar[:, :, :]

wcdf_divU_tilde = writeDS.createVariable(
    'divU_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_divU_tilde.long_name = "div u_tilde"
wcdf_divU_tilde.units = "s^-1"
wcdf_divU_tilde[:, :, :] = divU_tilde[:, :, :]


writeDS.close()
