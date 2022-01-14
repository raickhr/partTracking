from filteringFunctions import *
from netCDF4 import Dataset
import argparse

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

U = np.array(ds.variables['u'][:, :, :], dtype=np.float64)
V = np.array(ds.variables['v'][:, :, :], dtype=np.float64)
h = np.array(ds.variables['h'][:, :, :], dtype=np.float64)
P = np.array(ds.variables['e'][:, :, :], dtype=np.float64) * \
    9.81  # constant rho is omitted

timeLen, Ylen, Xlen = np.shape(U)
print('shape of the array U ', timeLen, Ylen, Xlen)
sys.stdout.flush()

farr = np.ones((Ylen, Xlen), dtype=np.float64)
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
omega_tilde_sq = omega_tilde ** 2
omega_bar = d_dx_V_bar - d_dy_U_bar

d_dx_omega_tilde_sq, d_dy_omega_tilde_sq = getGradient(
    omega_tilde_sq, dxInKm*1000, dyInKm*1000)
d_dt_omega_tilde_sq = (np.roll(omega_tilde_sq, -1, axis=0) - omega_tilde_sq)/dt
d_dt_omega_tilde_sq[tlen-1, :, :] = float('nan')

advecTermOmegaTildeSq = U_tilde * \
    d_dx_omega_tilde_sq + V_tilde * d_dy_omega_tilde_sq

## Dilation term
divUtilde = getDiv(U_tilde, V_tilde, dxInKm*1000, dyInKm*1000)
divfUtilde = getDiv(fU_tilde, fV_tilde, dxInKm*1000, dyInKm*1000)

R_Dilate = - h_bar * divUtilde * omega_tilde_sq - h_bar * omega_tilde * divfUtilde

## Baroclinic term
R_Barocl = omega_tilde * \
    (d_dx_h_bar*d_dy_hP_bar - d_dy_h_bar*d_dx_hP_bar)/(h_bar)

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

firstTerm = - omega_tilde * \
    getZCurl(divhTau_x_comp, divhTau_y_comp, dxInKm*1000, dyInKm*1000)
secondTerm = omega_tilde / h_bar * \
    (d_dx_h_bar * divhTau_y_comp - d_dy_h_bar * divhTau_x_comp)

R_substress = firstTerm + secondTerm


## LS_Drag

LS_Drag = omega_tilde * (d_dx_P_bar * d_dy_h_bar - d_dy_P_bar * d_dx_h_bar)

## SS_Drag
tau_P_gradh_xComp = Pdxh_bar - (P_bar*d_dx_h_bar)
tau_P_gradh_yComp = Pdyh_bar - (P_bar*d_dy_h_bar)

firstTerm = omega_tilde * \
    getZCurl(tau_P_gradh_xComp, tau_P_gradh_yComp,
             dxInKm*1000, dyInKm*1000)
secondTerm = - omega_tilde / \
    (h_bar) * (d_dx_h_bar * tau_P_gradh_yComp - d_dy_h_bar * tau_P_gradh_xComp)
SS_Drag = firstTerm + secondTerm


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

wcdf_Omega_tilde_sq = writeDS.createVariable(
    'Omega_tilde_sq', np.float64, ('Time', 'yh', 'xh'))
wcdf_Omega_tilde_sq.long_name = "Omega tilde squared"
wcdf_Omega_tilde_sq.units = "s^-4"
wcdf_Omega_tilde_sq[:, :, :] = omega_tilde_sq[:, :, :]

wcdf_d_dt_omega_tilde_sq = writeDS.createVariable(
    'd_dt_omega_tilde_sq', np.float64, ('Time', 'yh', 'xh'))
wcdf_d_dt_omega_tilde_sq.long_name = "d_dt Omega tilde squared"
wcdf_d_dt_omega_tilde_sq.units = "s^-3"
wcdf_d_dt_omega_tilde_sq[:, :, :] = d_dt_omega_tilde_sq[:, :, :]

wcdf_advecTermOmegaTildeSq = writeDS.createVariable(
    'advecTermOmegaTildeSq', np.float64, ('Time', 'yh', 'xh'))
wcdf_advecTermOmegaTildeSq.long_name = "advection term for Omega tilde squared"
wcdf_advecTermOmegaTildeSq.units = "s^-3"
wcdf_advecTermOmegaTildeSq[:, :, :] = advecTermOmegaTildeSq[:, :, :]

wcdf_R_Dilate = writeDS.createVariable(
    'R_Dilate', np.float64, ('Time', 'yh', 'xh'))
wcdf_R_Dilate.long_name = \
    "dilation term"
wcdf_R_Dilate.units = \
    "m s^-3"
wcdf_R_Dilate[:, :, :] = \
    R_Dilate[:, :, :]


wcdf_R_Barocl = writeDS.createVariable(
    'R_Barocl', np.float64, ('Time', 'yh', 'xh'))
wcdf_R_Barocl.long_name = \
    "Baroclinic term"
wcdf_R_Barocl.units = \
    "m s^-3"
wcdf_R_Barocl[:, :, :] = \
    R_Barocl[:, :, :]

wcdf_LS_Drag = writeDS.createVariable(
    'LS_Drag', np.float64, ('Time', 'yh', 'xh'))
wcdf_LS_Drag.long_name = \
    "Large Scale Drag Term"
wcdf_LS_Drag.units = \
    "m s^-3"
wcdf_LS_Drag[:, :, :] = \
    LS_Drag[:, :, :]


wcdf_SS_Drag = writeDS.createVariable(
    'SS_Drag', np.float64, ('Time', 'yh', 'xh'))
wcdf_SS_Drag.long_name = \
    "Small Scale Drag Term"
wcdf_SS_Drag.units = \
    "m s^-3"
wcdf_SS_Drag[:, :, :] = \
    SS_Drag[:, :, :]


wcdf_R_substress = writeDS.createVariable(
    'R_substress', np.float64, ('Time', 'yh', 'xh'))
wcdf_R_substress.long_name = \
    "Substress Term"
wcdf_R_substress.units = \
    "m s^-3"
wcdf_R_substress[:, :, :] = \
    R_substress[:, :, :]


writeDS.close()
