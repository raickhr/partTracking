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

parser.add_argument("--rho", "-r", type=float, default=1031.0, action='store',
                    help="this is the density of the layer")

args = parser.parse_args()

fileName = args.inputFile
fldLoc = args.fldLoc
rho = args.rho

f0 = 6.49e-05
beta = 2.0E-11

readFileName = fldLoc + '/' + fileName
fnum = ""
for s in fileName[0:8]:
    if s.isdigit():
        fnum += s
fnum = int(fnum)
nextReadFileName = fldLoc + \
    '/prog_{0:03d}'.format(fnum + 1) + "_RequiredFieldsOnly_4O.nc"
writeFileName = fldLoc + '/' + \
    fileName.replace("_RequiredFieldsOnly_4O.nc",
                     "_VorticityEqnTerms_4O.nc")

ds = Dataset(readFileName)

xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

dt = timeVal[1] - timeVal[0]
tlen = len(timeVal)

U = np.array(ds.variables['u'])
V = np.array(ds.variables['v'])
h = np.array(ds.variables['h'])
RV = np.array(ds.variables['RV'])
PV = np.array(ds.variables['PV'])


farr = np.ones(np.shape(U),dtype=np.float64)
(tlen, ylen, xlen) = np.shape(U)

for i in range(ylen):
    farr[:,i,:] = f0 + beta*yh[i]


d_dx_U, d_dy_U = getGradient(U, dxInKm*1000, dyInKm*1000)
d_dx_V, d_dy_V = getGradient(V, dxInKm*1000, dyInKm*1000)

omega = (d_dx_V - d_dy_U)
omegaSq = omega**2
totVort = omega + farr
divU = getDiv(U, V, dxInKm*1000, dyInKm*1000)

d_dx_omega, d_dy_omega = getGradient(omega, dxInKm*1000, dyInKm*1000)
d_dx_totVort, d_dy_totVort = getGradient(totVort, dxInKm*1000, dyInKm*1000)
d_dx_RV, d_dy_RV = getGradient(RV, dxInKm*1000, dyInKm*1000)
d_dx_PV, d_dy_PV = getGradient(PV, dxInKm*1000, dyInKm*1000)


d_dt_omega = (np.roll(omega,-1,axis = 0) - omega)/dt
d_dt_totVort = (np.roll(totVort,-1,axis = 0) - totVort)/dt
d_dt_RV = (np.roll(RV, -1, axis=0) - RV)/dt
d_dt_PV = (np.roll(PV, -1, axis=0) - PV)/dt

if path.exists(nextReadFileName):
    ds2 = Dataset(nextReadFileName)
    U_next = np.array(ds2.variables['u'][0:2, :, :])
    V_next = np.array(ds2.variables['v'][0:2, :, :])

    d_dx_U_next, d_dy_U_next = getGradient(U_next, dxInKm*1000, dyInKm*1000)
    d_dx_V_next, d_dy_V_next = getGradient(V_next, dxInKm*1000, dyInKm*1000)
    
    omega_next = (d_dx_V_next - d_dy_U_next)
    RV_next = np.array(ds2.variables['RV'][0:2, :, :])
    PV_next = np.array(ds2.variables['PV'][0:2, :, :])
    totVort_next = omega_next + farr[0:2, :, :]


    d_dt_omega[tlen-1, :, :] = (omega_next[0, :, :] - omega[tlen-1, :, :])/dt
    d_dt_RV[tlen-1, :, :] = (RV_next[0, :, :] - RV[tlen-1, :, :])/dt
    d_dt_PV[tlen-1, :, :] = (PV_next[0, :, :] - PV[tlen-1, :, :])/dt
    d_dt_totVort[tlen-1, :, :] = (totVort_next[0, :, :] - totVort[tlen-1, :, :])/dt

    ds2.close()

else:
    print('filling the last time derivative with nan value')
    d_dt_omega[tlen-1, :, :] = float('nan')
    d_dt_RV[tlen-1, :, :] = float('nan')
    d_dt_PV[tlen-1, :, :] = float('nan')
    d_dt_totVort[tlen-1, :, :] = float('nan')

advecTermOmega = U* d_dx_omega + V* d_dy_omega
advecTermTotVort = U * d_dx_totVort + V * d_dy_totVort
advecTermRV = U* d_dx_RV + V* d_dy_RV
advecTermPV = U* d_dx_PV + V* d_dy_PV

divTermOmega = omega * divU
divTermTotVort = totVort * divU
divTermRV = RV * divU
divTermPV = PV * divU

viscousTermOmega = 3.75e8 * (getBiharmonic(omega, dxInKm*1000, dyInKm*1000))
viscousTermTotVort = 3.75e8 * (getBiharmonic(totVort, dxInKm*1000, dyInKm*1000))
viscousTermRV = 3.75e8 * (getBiharmonic(RV, dxInKm*1000, dyInKm*1000))
viscousTermPV = 3.75e8 * (getBiharmonic(PV, dxInKm*1000, dyInKm*1000))


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


wcdf_u = writeDS.createVariable(
    'u', np.float64, ('Time', 'yh', 'xh'))
wcdf_u.long_name = "u"
wcdf_u.units = "m s^-1"
wcdf_u[:, :, :] = U[:, :, :]

wcdf_v = writeDS.createVariable(
    'v', np.float64, ('Time', 'yh', 'xh'))
wcdf_v.long_name = "v"
wcdf_v.units = "m s^-1"
wcdf_v[:, :, :] = V[:, :, :]

wcdf_omega = writeDS.createVariable(
    'omega', np.float64, ('Time', 'yh', 'xh'))
wcdf_omega.long_name = "omega"
wcdf_omega.units = "s^-1"
wcdf_omega[:, :, :] = omega[:, :, :]

wcdf_omegaSq = writeDS.createVariable(
    'omegaSq', np.float64, ('Time', 'yh', 'xh'))
wcdf_omegaSq.long_name = "omega"
wcdf_omegaSq.units = "s^-2"
wcdf_omegaSq[:, :, :] = omegaSq[:, :, :]

wcdf_totVort = writeDS.createVariable(
    'totVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_totVort.long_name = "total Vorticity"
wcdf_totVort.units = "s^-1"
wcdf_totVort[:, :, :] = totVort[:, :, :]

wcdf_RV = writeDS.createVariable(
    'RV', np.float64, ('Time', 'yh', 'xh'))
wcdf_RV.long_name = "RV from original output"
wcdf_RV.units = "s^-1"
wcdf_RV[:, :, :] = RV[:, :, :]

wcdf_PV = writeDS.createVariable(
    'PV', np.float64, ('Time', 'yh', 'xh'))
wcdf_PV.long_name = "PV from original output"
wcdf_PV.units = "s^-1"
wcdf_PV[:, :, :] = PV[:, :, :]


wcdf_d_dt_omega = writeDS.createVariable(
    'd_dt_omega', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_omega.long_name = "eulerian time derivative of omega"
wcdf_d_dt_omega.units = "s^-2"
wcdf_d_dt_omega[:, :, :] = d_dt_omega[:, :, :]

wcdf_d_dt_totVort = writeDS.createVariable(
    'd_dt_totVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_totVort.long_name = "eulerian time derivative of total Vorticity"
wcdf_d_dt_totVort.units = "s^-2"
wcdf_d_dt_totVort[:, :, :] = d_dt_totVort[:, :, :]

wcdf_d_dt_RV = writeDS.createVariable(
    'd_dt_RV', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_RV.long_name = "eulerian time derivative of RV from original output"
wcdf_d_dt_RV.units = "s^-2"
wcdf_d_dt_RV[:, :, :] = d_dt_RV[:, :, :]

wcdf_d_dt_PV = writeDS.createVariable(
    'd_dt_PV', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_PV.long_name = "eulerian time derivative of PV from original output"
wcdf_d_dt_PV.units = "s^-2"
wcdf_d_dt_PV[:, :, :] = d_dt_PV[:, :, :]


wcdf_advecTermOmega = writeDS.createVariable(
    'advecTermOmega', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermOmega.long_name = "advecTermOmega"
wcdf_advecTermOmega.units = "s^-2"
wcdf_advecTermOmega[:, :, :] = advecTermOmega[:, :, :]

wcdf_advecTermTotVort = writeDS.createVariable(
    'advecTermTotVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermTotVort.long_name = "advecTermTotVort"
wcdf_advecTermTotVort.units = "s^-2"
wcdf_advecTermTotVort[:, :, :] = advecTermTotVort[:, :, :]

wcdf_advecTermRV = writeDS.createVariable(
    'advecTermRV', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermRV.long_name = "advecTermRV"
wcdf_advecTermRV.units = "s^-2"
wcdf_advecTermRV[:, :, :] = advecTermRV[:, :, :]

wcdf_advecTermPV = writeDS.createVariable(
    'advecTermPV', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermPV.long_name = "advecTermPV"
wcdf_advecTermPV.units = "s^-2"
wcdf_advecTermPV[:, :, :] = advecTermPV[:, :, :]


wcdf_divTermOmega = writeDS.createVariable(
    'divTermOmega', np.float64, ('Time', 'yh', 'xh'))
wcdf_divTermOmega.long_name = "divTermOmega"
wcdf_divTermOmega.units = "s^-2"
wcdf_divTermOmega[:, :, :] = divTermOmega[:, :, :]

wcdf_divTermTotVort = writeDS.createVariable(
    'divTermTotVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_divTermTotVort.long_name = "divTermTotVort"
wcdf_divTermTotVort.units = "s^-2"
wcdf_divTermTotVort[:, :, :] = divTermTotVort[:, :, :]

wcdf_divTermRV = writeDS.createVariable(
    'divTermRV', np.float64, ('Time', 'yh', 'xh'))
wcdf_divTermRV.long_name = "divTermRV"
wcdf_divTermRV.units = "s^-2"
wcdf_divTermRV[:, :, :] = divTermRV[:, :, :]

wcdf_divTermPV = writeDS.createVariable(
    'divTermPV', np.float64, ('Time', 'yh', 'xh'))
wcdf_divTermPV.long_name = "divTermPV"
wcdf_divTermPV.units = "s^-2"
wcdf_divTermPV[:, :, :] = divTermPV[:, :, :]

wcdf_viscousTermOmega = writeDS.createVariable(
    'viscousTermOmega', np.float64, ('Time', 'yh', 'xh'))
wcdf_viscousTermOmega.long_name = "viscousTermOmega"
wcdf_viscousTermOmega.units = "s^-2"
wcdf_viscousTermOmega[:, :, :] = viscousTermOmega[:, :, :]

wcdf_viscousTermTotVort = writeDS.createVariable(
    'viscousTermTotVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_viscousTermTotVort.long_name = "divTermTotVort"
wcdf_viscousTermTotVort.units = "s^-2"
wcdf_viscousTermTotVort[:, :, :] = viscousTermTotVort[:, :, :]

wcdf_viscousTermRV = writeDS.createVariable(
    'viscousTermRV', np.float64, ('Time', 'yh', 'xh'))
wcdf_viscousTermRV.long_name = "divTermRV"
wcdf_viscousTermRV.units = "s^-2"
wcdf_viscousTermRV[:, :, :] = divTermRV[:, :, :]

wcdf_viscousTermPV = writeDS.createVariable(
    'viscousTermPV', np.float64, ('Time', 'yh', 'xh'))
wcdf_viscousTermPV.long_name = "viscousTermPV"
wcdf_viscousTermPV.units = "s^-2"
wcdf_viscousTermPV[:, :, :] = viscousTermPV[:, :, :]

writeDS.close()
