from filteringFunctions import *
from netCDF4 import Dataset
import argparse

from pythonFiles.writeFilteredFieldsOrig import P_bar

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
writeFileName = fldLoc + '/' + \
    fileName.replace("_RequiredFieldsOnly_4FilteredFields.nc","_LambdaAndPiValues_4O.nc")

ds = Dataset(readFileName)

xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
timeVal = np.array(ds.variables['Time'])
timeUnits = ds.variables['Time'].units

U_bar = np.array(ds.variables['u_bar'])
V_bar = np.array(ds.variables['v_bar'])
hU_bar = np.array(ds.variables['hu_bar'])
hV_bar = np.array(ds.variables['hv_bar'])
hUU_bar = np.array(ds.variables['huu_bar'])
hUV_bar = np.array(ds.variables['huv_bar'])
hVV_bar = np.array(ds.variables['hvv_bar'])
h_bar = np.array(ds.variables['h_bar'])
hP_bar = np.array(ds.variables['hp_bar'])
P_bar = np.array(ds.variables['p_Bar'])
Pdxh_bar = np.array(ds.variables['pdx_h_bar'])
Pdyh_bar = np.array(ds.variables['pdy_h_bar'])

U_tilde = hU_bar/h_bar
V_tilde = hV_bar/h_bar

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

Pi = -h_bar *(d_dx_U_tilde * (UU_tilde - U_tilde**2) +
              d_dy_U_tilde * (UV_tilde - U_tilde * V_tilde) +
              d_dx_V_tilde * (UV_tilde - U_tilde * V_tilde) +
              d_dy_V_tilde * (VV_tilde - V_tilde**2))

Lambda = 1/(h_bar * rho) * (d_dx_hP_bar * (hU_bar - U_bar * h_bar) +
                    d_dy_hP_bar * (hV_bar - V_bar * h_bar))


Jx = 1/rho * U_bar * hP_bar + U_tilde * h_bar * ((UU_tilde - U_tilde*U_tilde)) +\
                      V_tilde * h_bar * ((UV_tilde - V_tilde*U_tilde))

Jy = 1/rho * V_bar * hP_bar + V_tilde * h_bar * ((VV_tilde - V_tilde*U_tilde)) +\
                      U_tilde * h_bar * ((UV_tilde - V_tilde*U_tilde))

nablaJ = getDiv(Jx, Jy, dxInKm*1000, dyInKm*1000)

divUbar = getDiv(U_bar, V_bar, dxInKm*1000, dyInKm*1000)
pDilation = 1/rho * hP_bar * divUbar

tau_P_gradh_xComp = Pdxh_bar - (P_bar*d_dx_h_bar)
tau_P_gradh_yComp = Pdyh_bar - (P_bar*d_dy_h_bar)

engSmScFormDrag = 1/rho * (U_tilde * tau_P_gradh_xComp + V_tilde * tau_P_gradh_yComp)
engLgScFormDrag = 1/rho * (U_tilde * d_dx_P_bar + V_tilde * d_dy_P_bar)

omega_tilde = d_dx_V_tilde - d_dy_U_tilde
omega_tilde_sq = omega_tilde **2

omega_bar = d_dx_V_bar - d_dy_U_bar

S_tilde_11 = d_dx_U_tilde
S_tilde_22 = d_dy_V_tilde
S_tilde_12 = 0.5 * (d_dx_V_tilde + d_dy_U_tilde)

S_tilde_sq = S_tilde_11**2 + S_tilde_22**2 + 2 * S_tilde_12**2

S_bar_11 = d_dx_U_bar
S_bar_22 = d_dy_V_bar
S_bar_12 = 0.5 * (d_dx_V_bar + d_dy_U_bar)

S_bar_sq = S_bar_11**2 + S_bar_22**2 + 2*S_bar_12**2
d_dx_h_bar, d_dy_h_bar = getGradient(h_bar, dxInKm*1000, dyInKm*1000)

C = 0.125

Lambda_str = d_dx_h_bar * S_bar_11 * d_dx_hP_bar + \
    d_dx_h_bar * S_bar_12 * d_dy_hP_bar + \
    d_dy_h_bar * S_bar_12 * d_dx_hP_bar + \
    d_dy_h_bar * S_bar_22 * d_dy_hP_bar

Lambda_str *= C * (ellInKm*1000)**2 * 1/(rho * h_bar)

Lambda_rot = 0.5 * C * (ellInKm*1000)**2 * omega_bar * \
    (d_dx_h_bar*d_dy_hP_bar - d_dy_h_bar*d_dx_hP_bar)/(rho * h_bar)


S_Baro = (d_dx_h_bar * S_tilde_11 * d_dx_hP_bar +
          d_dx_h_bar * S_tilde_12 * d_dy_hP_bar +
          d_dy_h_bar * S_tilde_12 * d_dx_hP_bar +
          d_dy_h_bar * S_tilde_22 * d_dy_hP_bar) / (rho * h_bar)

R_Barocl = omega_tilde * \
    (d_dx_h_bar*d_dy_hP_bar - d_dy_h_bar*d_dx_hP_bar)/(rho * h_bar)


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

wcdf_U_tilde = writeDS.createVariable(
    'u_tilde', np.float32, ('Time', 'yh', 'xh'))
wcdf_U_tilde.long_name = "u_tilde"
wcdf_U_tilde.units = "m s^-1"
wcdf_U_tilde[:, :, :] = U_tilde[:, :, :]

wcdf_V_tilde = writeDS.createVariable(
    'v_tilde', np.float32, ('Time', 'yh', 'xh'))
wcdf_V_tilde.long_name = "v_tilde"
wcdf_V_tilde.units = "m s^-1"
wcdf_V_tilde[:, :, :] = V_tilde[:, :, :]

wcdf_h_bar = writeDS.createVariable(
    'h_bar', np.float32, ('Time', 'yh', 'xh'))
wcdf_h_bar.long_name = "h_bar"
wcdf_h_bar.units = "m"
wcdf_h_bar[:, :, :] = h_bar[:, :, :]

wcdf_omega_tilde = writeDS.createVariable(
    'omega_tilde', np.float32, ('Time', 'yh', 'xh'))
wcdf_omega_tilde.long_name = "weighted filtered omega"
wcdf_omega_tilde.units = "s^-1"
wcdf_omega_tilde[:, :, :] = omega_tilde[:, :, :]

wcdf_Omega_tilde_sq = writeDS.createVariable(
    'Omega_tilde_sq', np.float32, ('Time', 'yh', 'xh'))
wcdf_Omega_tilde_sq.long_name = "Omega tilde squared"
wcdf_Omega_tilde_sq.units = "s^-4"
wcdf_Omega_tilde_sq[:, :, :] = omega_tilde_sq[:,:,:]


wcdf_Pi = writeDS.createVariable('Pi', np.float32, ('Time', 'yh', 'xh'))
wcdf_Pi.long_name = "Energy Cascade Term, Pi"
wcdf_Pi.units = "m^3 s^-3"
wcdf_Pi[:, :, :] = Pi[:, :, :]


wcdf_Lambda = writeDS.createVariable(
    'Lambda', np.float32, ('Time', 'yh', 'xh'))
wcdf_Lambda.long_name = "Energy Cascade Term, Lambda"
wcdf_Lambda.units = "m^3 s^-3"
wcdf_Lambda[:, :, :] = Lambda[:, :, :]

wcdf_nablaJ = writeDS.createVariable(
    'nablaJ', np.float32, ('Time', 'yh', 'xh'))
wcdf_nablaJ.long_name = \
    "Transport term"
wcdf_nablaJ.units = \
    "m^3 s^-3"
wcdf_nablaJ[:, :, :] = \
     nablaJ[:, :, :]

wcdf_pDilation = writeDS.createVariable(
    'pDilation', np.float32, ('Time', 'yh', 'xh'))
wcdf_pDilation.long_name = \
    "Pressure Dilation"
wcdf_pDilation.units = \
    "m^3 s^-3"
wcdf_pDilation[:, :, :] = \
    pDilation[:, :, :]

wcdf_engLgScFormDrag = writeDS.createVariable(
    'engLgScFormDrag', np.float32, ('Time', 'yh', 'xh'))
wcdf_engLgScFormDrag.long_name = \
    "energy from Large Scale Form Drag"
wcdf_engLgScFormDrag.units = \
    "m^3 s^-3"
wcdf_engLgScFormDrag[:, :, :] = \
    engLgScFormDrag[:, :, :]

wcdf_engSmScFormDrag = writeDS.createVariable(
    'engSmScFormDrag', np.float32, ('Time', 'yh', 'xh'))
wcdf_engSmScFormDrag.long_name = \
    "energy from Small Scale Form Drag"
wcdf_engSmScFormDrag.units = \
    "m^3 s^-3"
wcdf_engSmScFormDrag[:, :, :] = \
    engSmScFormDrag[:, :, :]


wcdf_Lambda_str = writeDS.createVariable(
    'Lambda_str', np.float32, ('Time', 'yh', 'xh'))
wcdf_Lambda_str.long_name = "Modelled Energy Cascade Term (Strain Part), Lambda"
wcdf_Lambda_str.units = "m^-1 s^-3"
wcdf_Lambda_str[:, :, :] = Lambda_str[:, :, :]

wcdf_Lambda_rot = writeDS.createVariable(
    'Lambda_rot', np.float32, ('Time', 'yh', 'xh'))
wcdf_Lambda_rot.long_name = "Modelled Energy Cascade Term (Rotation Part), Lambda"
wcdf_Lambda_rot.units = "m^-1 s^-3"
wcdf_Lambda_rot[:, :, :] = Lambda_rot[:, :, :]

wcdf_R_Barocl = writeDS.createVariable(
    'R_Barocl', np.float32, ('Time', 'yh', 'xh'))
wcdf_R_Barocl.long_name = "R_baroclinic, a source term from evolution of omega squared"
wcdf_R_Barocl.units = "kg m^-3 s^-1"
wcdf_R_Barocl[:, :, :] = R_Barocl[:, :, :]

wcdf_S_Baro = writeDS.createVariable(
    'S_Baro', np.float32, ('Time', 'yh', 'xh'))
wcdf_S_Baro.long_name = "S_barotropic, a source term from evolution of S squared"
wcdf_S_Baro.units = "kg m^-3 s^-1"
wcdf_S_Baro[:, :, :] = S_Baro[:, :, :]

wcdf_S_tilde_sq = writeDS.createVariable(
    'S_tilde_sq', np.float32, ('Time', 'yh', 'xh'))
wcdf_S_tilde_sq.long_name = "S_tilde_squared SijSij"
wcdf_S_tilde_sq.units = "s^-4"
wcdf_S_tilde_sq[:, :, :] = S_tilde_sq[:, :, :]


writeDS.close()
