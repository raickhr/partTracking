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


writeSuffix = '_PotVortEqnTerms_{0:03d}km_4O.nc'.format(ellInKm)
writeFileName = fldLoc + '/' + \
    fileName.replace(readSuffix, writeSuffix)

ds = Dataset(readFileName)


xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

Xlen = len(xh)
Ylen = len(yh)

dt = timeVal[1] - timeVal[0]
tlen = len(timeVal)

farr = np.ones((tlen, Ylen, Xlen), dtype=np.float64)
f0 = 6.49e-05
beta = 2.0E-11

for i in range(Ylen):
    farr[:, i, :] = f0 + beta*yh[i]

U_bar = np.array(ds.variables['u_bar'])
V_bar = np.array(ds.variables['v_bar'])
hfU_bar = np.array(ds.variables['hfu_bar'])
hfV_bar = np.array(ds.variables['hfv_bar'])
hU_bar = np.array(ds.variables['hu_bar'])
hV_bar = np.array(ds.variables['hv_bar'])
hUU_bar = np.array(ds.variables['huu_bar'])
hUV_bar = np.array(ds.variables['huv_bar'])
hVV_bar = np.array(ds.variables['hvv_bar'])
h_bar = np.array(ds.variables['h_bar'])
hP_bar = np.array(ds.variables['hp_bar'])
P_bar = np.array(ds.variables['p_bar'])
Pdxh_bar = np.array(ds.variables['pdx_h_bar'])
Pdyh_bar = np.array(ds.variables['pdy_h_bar'])

U_tilde = hU_bar/h_bar
V_tilde = hV_bar/h_bar

fU_tilde = hfU_bar/h_bar
fV_tilde = hfV_bar/h_bar

UU_tilde = hUU_bar/h_bar
UV_tilde = hUV_bar/h_bar
VV_tilde = hVV_bar/h_bar

d_dx_U_tilde, d_dy_U_tilde = getGradient(U_tilde, dxInKm*1000, dyInKm*1000)
d_dx_V_tilde, d_dy_V_tilde = getGradient(V_tilde, dxInKm*1000, dyInKm*1000)
d_dx_hP_bar, d_dy_hP_bar = getGradient(hP_bar, dxInKm*1000, dyInKm*1000)
d_dx_P_bar, d_dy_P_bar = getGradient(P_bar, dxInKm*1000, dyInKm*1000)
d_dx_h_bar, d_dy_h_bar = getGradient(h_bar, dxInKm*1000, dyInKm*1000)

d_dx_U_bar, d_dy_U_bar = getGradient(U_bar, dxInKm*1000, dyInKm*1000)
d_dx_V_bar, d_dy_V_bar = getGradient(V_bar, dxInKm*1000, dyInKm*1000)


omega_tilde = d_dx_V_tilde - d_dy_U_tilde
omega_bar = d_dx_V_bar - d_dy_U_bar
potVort = (omega_tilde +  farr)/h_bar

d_dt_potVort = (np.roll(potVort, -1, axis=0) - potVort)/dt

if path.exists(nextReadFileName):
    ds2 = Dataset(nextReadFileName)
    hU_bar_next = np.array(ds2.variables['hu_bar'][0:2, :, :])
    hV_bar_next = np.array(ds2.variables['hv_bar'][0:2, :, :])
    h_bar_next = np.array(ds2.variables['h_bar'][0:2, :, :])
    U_tilde_next = hU_bar_next/h_bar_next
    V_tilde_next = hV_bar_next/h_bar_next
    d_dx_U_tilde_next, d_dy_U_tilde_next = getGradient(
        U_tilde_next, dxInKm*1000, dyInKm*1000)
    d_dx_V_tilde_next, d_dy_V_tilde_next = getGradient(
        V_tilde_next, dxInKm*1000, dyInKm*1000)
    omega_tilde_next = d_dx_V_tilde_next - d_dy_U_tilde_next
    potVort_next = (omega_tilde_next + farr[0:2, :, :])/h_bar_next
    d_dt_potVort[tlen-1, :,
                        :] = (potVort_next[0, :, :] - potVort[tlen-1, :, :])/dt
    ds2.close()

else:
    print('filling the last time derivative with nan value')
    d_dt_potVort[tlen-1, :, :] = float('nan')


d_dx_potVort, d_dy_potVort = getGradient(
    potVort, dxInKm*1000, dyInKm*1000)
advecTermPotVort = U_tilde * \
    d_dx_potVort + V_tilde * d_dy_potVort

## Baroclinic term
R_Barocl = (d_dx_h_bar*d_dy_hP_bar - d_dy_h_bar*d_dx_hP_bar)/(h_bar**3)

## LS_Drag

LS_Drag = (d_dx_P_bar * d_dy_h_bar - d_dy_P_bar * d_dx_h_bar)/(h_bar **2)

## SS_Drag
tau_P_gradh_xComp = Pdxh_bar - (P_bar*d_dx_h_bar)
tau_P_gradh_yComp = Pdyh_bar - (P_bar*d_dy_h_bar)

firstTerm = getZCurl(tau_P_gradh_xComp, tau_P_gradh_yComp,
             dxInKm*1000, dyInKm*1000)/(h_bar**2)
secondTerm = -(d_dx_h_bar * tau_P_gradh_yComp - d_dy_h_bar * tau_P_gradh_xComp)/(h_bar**3)
SS_Drag = firstTerm + secondTerm


## R_Substress term
hbar_TauTildeUU = h_bar * (UU_tilde - U_tilde * U_tilde)
hbar_TauTildeUV = h_bar * (UV_tilde - U_tilde * V_tilde)
hbar_TauTildeVV = h_bar * (VV_tilde - V_tilde * V_tilde)

d_dx_hbar_TauTildeUU, d_dy_hbar_TauTildeUU = getGradient(
    hbar_TauTildeUU, dxInKm*1000, dyInKm*1000)
d_dx_hbar_TauTildeUV, d_dy_hbar_TauTildeUV = getGradient(
    hbar_TauTildeUV, dxInKm*1000, dyInKm*1000)
d_dx_hbar_TauTildeVV, d_dy_hbar_TauTildeVV = getGradient(
    hbar_TauTildeVV, dxInKm*1000, dyInKm*1000)

divhTau_x_comp = d_dx_hbar_TauTildeUU + d_dy_hbar_TauTildeUV
divhTau_y_comp = d_dx_hbar_TauTildeUV + d_dy_hbar_TauTildeVV

firstTerm = - getZCurl(divhTau_x_comp, divhTau_y_comp, dxInKm*1000, dyInKm*1000)/(h_bar**2)
secondTerm = (d_dx_h_bar * divhTau_y_comp - d_dy_h_bar * divhTau_x_comp)/(h_bar**3)

R_substress = firstTerm + secondTerm




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

wcdf_h_bar = writeDS.createVariable(
    'h_bar', np.float64, ('Time', 'yh', 'xh'))
wcdf_h_bar.long_name = "h_bar"
wcdf_h_bar.units = "m"
wcdf_h_bar[:, :, :] = h_bar[:, :, :]

wcdf_omega_tilde = writeDS.createVariable(
    'omega_tilde', np.float64, ('Time', 'yh', 'xh'))
wcdf_omega_tilde.long_name = "weighted filtered omega"
wcdf_omega_tilde.units = "s^-1"
wcdf_omega_tilde[:, :, :] = omega_tilde[:, :, :]

wcdf_potVort = writeDS.createVariable(
    'potVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_potVort.long_name = "Potential Vorticity"
wcdf_potVort.units = "s^-4"
wcdf_potVort[:, :, :] = potVort[:, :, :]

wcdf_d_dt_potVort = writeDS.createVariable(
    'd_dt_potVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_potVort.long_name = "d_dt Potential Vorticity"
wcdf_d_dt_potVort.units = "m^-1 s^-2"
wcdf_d_dt_potVort[:, :, :] = d_dt_potVort[:, :, :]

wcdf_advecTermPotVort = writeDS.createVariable(
    'advecTermPotVort', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermPotVort.long_name = "advection term for Potential Vorticity"
wcdf_advecTermPotVort.units = "s^-3"
wcdf_advecTermPotVort[:, :, :] = advecTermPotVort[:, :, :]


wcdf_R_Barocl = writeDS.createVariable(
    'R_Barocl', np.float64, ('Time', 'yh', 'xh'))
wcdf_R_Barocl.long_name = \
    "Baroclinic term"
wcdf_R_Barocl.units = \
    "m^-1 s^-2"
wcdf_R_Barocl[:, :, :] = \
    R_Barocl[:, :, :]

wcdf_LS_Drag = writeDS.createVariable(
    'LS_Drag', np.float64, ('Time', 'yh', 'xh'))
wcdf_LS_Drag.long_name = \
    "Large Scale Drag Term"
wcdf_LS_Drag.units = \
    "m^-1 s^-2"
wcdf_LS_Drag[:, :, :] = \
    LS_Drag[:, :, :]


wcdf_SS_Drag = writeDS.createVariable(
    'SS_Drag', np.float64, ('Time', 'yh', 'xh'))
wcdf_SS_Drag.long_name = \
    "Small Scale Drag Term"
wcdf_SS_Drag.units = \
    "m^-1 s^-2"
wcdf_SS_Drag[:, :, :] = \
    SS_Drag[:, :, :]


wcdf_R_substress = writeDS.createVariable(
    'R_substress', np.float64, ('Time', 'yh', 'xh'))
wcdf_R_substress.long_name = \
    "Substress Term"
wcdf_R_substress.units = \
    "m^-1 s^-2"
wcdf_R_substress[:, :, :] = \
    R_substress[:, :, :]


writeDS.close()
