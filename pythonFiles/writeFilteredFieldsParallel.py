from mpi4py import MPI
from filteringFunctions import *
import numpy as np
from netCDF4 import Dataset
import argparse
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


dxInKm = 5
dyInKm = 5

if rank == 0:
    print("running with {0:d} processors".format( nprocs))

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
        fileName.replace('_RequiredFieldsOnly_4O.nc', '_FilteredFields_4O.nc')

    ds = Dataset(readFileName)

    xh = np.array(ds.variables['xh'])
    yh = np.array(ds.variables['yh'])
    timeVal = np.array(ds.variables['Time'])
    timeUnits = ds.variables['Time'].units


    globalU = np.array(ds.variables['u'][:,:,:], dtype = float)
    globalV = np.array(ds.variables['v'][:,:,:], dtype = float)
    globalh = np.array(ds.variables['h'][:,:,:], dtype = float)
    globalP = np.array(ds.variables['e'][:,:,:], dtype = float) * 9.81  # a constant factor of rho is omitted

    timeLen, Ylen, Xlen = np.shape(globalU)
    print('File reading complete by processor', rank)
    print('shape of the array U ', timeLen, Ylen, Xlen)
    sys.stdout.flush()

    farr = np.ones((Ylen, Xlen), dtype=float)
    f0 = 6.49e-05
    beta = 2.0E-11
    
    y = yh - np.mean(yh)

    for i in range(Ylen):
        farr[i, :] = f0 + beta*y[i]

    localTimeLen = timeLen//nprocs
    extra = timeLen%nprocs
    localTimeLenList = [localTimeLen] * nprocs
    split_size = [0]* nprocs
    count = 0 

    if extra >0 :
        print("All processors do not have same size data. ")
        while extra > 0:
            localTimeLenList[count]+= 1
            extra -= 1
            count += 1

    
    print("Division of work is with time dimension")
    sys.stdout.flush()

    for i in range(nprocs):
        print('Processor :', i, 'Time dimension size', localTimeLenList[i])
        split_size[i] = localTimeLenList[i] * Ylen * Xlen

    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]
    sys.stdout.flush()

else:
    globalU = None
    globalV = None
    globalh = None
    globalP = None
    farr = None

    timeLen = None
    Xlen = None
    Ylen = None
    localTimeLenList = None
    localTimeLen = None

    ellInKm = None
    split_size = None
    split_disp = None

comm.Barrier()
timeLen = comm.bcast(timeLen, root = 0)
Xlen = comm.bcast(Xlen, root=0)
Ylen = comm.bcast(Ylen, root=0)
farr = comm.bcast(farr, root=0)
ellInKm = comm.bcast(ellInKm, root = 0)
split_size = comm.bcast(split_size, root=0)
split_disp = comm.bcast(split_disp, root=0)

print('arrays broadcast complete')
sys.stdout.flush()


localTimeLen = comm.scatter(localTimeLenList, root=0)
print('scattering the local Time dimension size complete')
sys.stdout.flush()

localU = np.zeros((localTimeLen, Ylen, Xlen), dtype=float)
localV = np.zeros((localTimeLen, Ylen, Xlen), dtype=float)
localh = np.zeros((localTimeLen, Ylen, Xlen), dtype=float)
localP = np.zeros((localTimeLen, Ylen, Xlen), dtype=float)

comm.Scatterv([globalU, split_size, split_disp, MPI.DOUBLE], localU, root=0)
comm.Scatterv([globalV, split_size, split_disp, MPI.DOUBLE], localV, root=0)
comm.Scatterv([globalh, split_size, split_disp, MPI.DOUBLE], localh, root=0)
comm.Scatterv([globalP, split_size, split_disp, MPI.DOUBLE], localP, root=0)

print('arrays scattering complete')
sys.stdout.flush()

## f*U 
farr = np.repeat(farr[np.newaxis, :, :], localTimeLen, axis=0)
flocalU = localU * farr
flocalV = localV * farr

#Velocity gradients calculations
localdx_U, localdy_U = getGradient(localU, dxInKm*1000, dyInKm*1000)
localdx_V, localdy_V = getGradient(localV, dxInKm*1000, dyInKm*1000)

localdx_h, localdy_h = getGradient(localh, dxInKm*1000, dyInKm*1000)

U_bar = get_filtered_Field(localU ,  ellInKm, dxInKm, dyInKm)
V_bar = get_filtered_Field(localV ,  ellInKm, dxInKm, dyInKm)
hfU_bar = get_filtered_Field(localh * flocalU,  ellInKm, dxInKm, dyInKm)
hfV_bar = get_filtered_Field(localh * flocalV,  ellInKm, dxInKm, dyInKm)
UU_bar = get_filtered_Field(localU * localU ,  ellInKm, dxInKm, dyInKm)
VV_bar = get_filtered_Field(localV * localV ,  ellInKm, dxInKm, dyInKm)
UV_bar = get_filtered_Field(localU * localV ,  ellInKm, dxInKm, dyInKm)
hU_bar = get_filtered_Field(localh * localU ,  ellInKm, dxInKm, dyInKm)
hV_bar = get_filtered_Field(localh * localV ,  ellInKm, dxInKm, dyInKm)
hUU_bar = get_filtered_Field(localh * localU * localU ,  ellInKm, dxInKm, dyInKm)
hUV_bar = get_filtered_Field(localh * localU * localV ,  ellInKm, dxInKm, dyInKm)
hVV_bar = get_filtered_Field(localh * localV * localV ,  ellInKm, dxInKm, dyInKm)
h_bar = get_filtered_Field(localh ,  ellInKm, dxInKm, dyInKm)
P_bar = get_filtered_Field(localP ,  ellInKm, dxInKm, dyInKm)
hP_bar = get_filtered_Field(localh * localP ,  ellInKm, dxInKm, dyInKm)
Pdx_h_bar = get_filtered_Field(localP * localdx_h, ellInKm, dxInKm, dyInKm)
Pdy_h_bar = get_filtered_Field(localP * localdy_h, ellInKm, dxInKm, dyInKm)


if rank == 0: 
    global_U_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_V_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hfU_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hfV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_UU_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_VV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_UV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hU_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hUU_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hUV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hVV_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_h_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_P_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_hP_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_Pdx_h_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
    global_Pdy_h_bar = np.zeros((timeLen, Ylen, Xlen), dtype=float)
else:
    global_U_bar = None
    global_V_bar = None
    global_hfU_bar = None
    global_hfV_bar = None
    global_UU_bar = None
    global_VV_bar = None
    global_UV_bar = None
    global_hU_bar = None
    global_hV_bar = None
    global_hUU_bar = None
    global_hUV_bar = None
    global_hVV_bar = None
    global_h_bar = None
    global_P_bar = None
    global_hP_bar = None
    global_Pdx_h_bar = None
    global_Pdy_h_bar = None


print('calculation complete by rank', rank)
sys.stdout.flush()

comm.Gatherv(sendbuf=U_bar, recvbuf = (global_U_bar, split_size), root=0)
comm.Gatherv(sendbuf=V_bar, recvbuf = (global_V_bar, split_size), root=0)
comm.Gatherv(sendbuf=hfU_bar, recvbuf=(global_fU_bar, split_size), root=0)
comm.Gatherv(sendbuf=hfV_bar, recvbuf=(global_fV_bar, split_size), root=0)
comm.Gatherv(sendbuf=UU_bar, recvbuf = (global_UU_bar, split_size), root=0)
comm.Gatherv(sendbuf=VV_bar, recvbuf = (global_VV_bar, split_size), root=0)
comm.Gatherv(sendbuf=UV_bar, recvbuf = (global_UV_bar, split_size), root=0)
comm.Gatherv(sendbuf=hU_bar, recvbuf = (global_hU_bar, split_size), root=0)
comm.Gatherv(sendbuf=hV_bar, recvbuf = (global_hV_bar, split_size), root=0)
comm.Gatherv(sendbuf=hUU_bar, recvbuf = (global_hUU_bar, split_size), root=0)
comm.Gatherv(sendbuf=hUV_bar, recvbuf = (global_hUV_bar, split_size), root=0)
comm.Gatherv(sendbuf=hVV_bar, recvbuf = (global_hVV_bar, split_size), root=0)
comm.Gatherv(sendbuf=h_bar, recvbuf = (global_h_bar, split_size), root=0)
comm.Gatherv(sendbuf=P_bar, recvbuf = (global_P_bar, split_size), root=0)
comm.Gatherv(sendbuf=hP_bar, recvbuf = (global_hP_bar, split_size), root=0)
comm.Gatherv(sendbuf=Pdx_h_bar, recvbuf = (global_Pdx_h_bar, split_size), root=0)
comm.Gatherv(sendbuf=Pdy_h_bar, recvbuf = (global_Pdy_h_bar, split_size), root=0)


if rank == 0:
    print('Gathering complete.')
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
    wcdf_U_bar[:, :, :] = global_U_bar[:, :, :]

    wcdf_V_bar = writeDS.createVariable(
        'v_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_V_bar.long_name = "v_bar"
    wcdf_V_bar.units = "m s^-1"
    wcdf_V_bar[:, :, :] = global_V_bar[:, :, :]

    wcdf_fU_bar = writeDS.createVariable(
        'fu_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_fU_bar.long_name = "fu_bar"
    wcdf_fU_bar.units = "m s^-2"
    wcdf_fU_bar[:, :, :] = global_fU_bar[:, :, :]

    wcdf_fV_bar = writeDS.createVariable(
        'fv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_fV_bar.long_name = "v_bar"
    wcdf_fV_bar.units = "m s^-2"
    wcdf_fV_bar[:, :, :] = global_fV_bar[:, :, :]

    wcdf_h_bar = writeDS.createVariable(
        'h_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_h_bar.long_name = "h_bar"
    wcdf_h_bar.units = "m"
    wcdf_h_bar[:, :, :] = global_h_bar[:, :, :]

    wcdf_hU_bar = writeDS.createVariable(
        'hu_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hU_bar.long_name = "hu_bar"
    wcdf_hU_bar.units = "m^2 s^-1"
    wcdf_hU_bar[:, :, :] = global_hU_bar[:, :, :]

    wcdf_hV_bar = writeDS.createVariable(
        'hv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hV_bar.long_name = "hv_bar"
    wcdf_hV_bar.units = "m^2 s^-1"
    wcdf_hV_bar[:, :, :] = global_hV_bar[:, :, :]

    wcdf_hUU_bar = writeDS.createVariable(
        'huu_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hUU_bar.long_name = "huu_bar"
    wcdf_hUU_bar.units = "m^3 s^-2"
    wcdf_hUU_bar[:, :, :] = global_hUU_bar[:, :, :]

    wcdf_hVV_bar = writeDS.createVariable(
        'hvv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hVV_bar.long_name = "hvv_bar"
    wcdf_hVV_bar.units = "m^3 s^-2"
    wcdf_hVV_bar[:, :, :] = global_hVV_bar[:, :, :]

    wcdf_hUV_bar = writeDS.createVariable(
        'huv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hUV_bar.long_name = "huv_bar"
    wcdf_hUV_bar.units = "m^3 s^-2"
    wcdf_hUV_bar[:, :, :] = global_hUV_bar[:, :, :]

    wcdf_UU_bar = writeDS.createVariable(
        'uu_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_UU_bar.long_name = "uu_bar"
    wcdf_UU_bar.units = "m^2 s^-2"
    wcdf_UU_bar[:, :, :] = global_UU_bar[:, :, :]

    wcdf_VV_bar = writeDS.createVariable(
        'vv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_VV_bar.long_name = "vv_bar"
    wcdf_VV_bar.units = "m^2 s^-2"
    wcdf_VV_bar[:, :, :] = global_VV_bar[:, :, :]

    wcdf_UV_bar = writeDS.createVariable(
        'uv_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_UV_bar.long_name = "uv_bar"
    wcdf_UV_bar.units = "m^2 s^-2"
    wcdf_UV_bar[:, :, :] = global_UV_bar[:, :, :]

    wcdf_P_bar = writeDS.createVariable(
        'p_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_P_bar.long_name = "p_bar"
    wcdf_P_bar.units = "m^2 s^-2"
    wcdf_P_bar[:, :, :] = global_P_bar[:, :, :]

    wcdf_Pdx_h_bar = writeDS.createVariable(
        'pdx_h_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_Pdx_h_bar.long_name = "pdx_bar"
    wcdf_Pdx_h_bar.units = "m^2 s^-2"
    wcdf_Pdx_h_bar[:, :, :] = global_Pdx_h_bar[:, :, :]

    wcdf_Pdy_h_bar = writeDS.createVariable(
        'pdy_h_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_Pdy_h_bar.long_name = "pdy_bar"
    wcdf_Pdy_h_bar.units = "m^2 s^-2"
    wcdf_Pdy_h_bar[:, :, :] = global_Pdy_h_bar[:, :, :]

    wcdf_hP_bar = writeDS.createVariable(
        'hp_bar', np.float32, ('Time', 'yh', 'xh'))
    wcdf_hP_bar.long_name = "hp_bar"
    wcdf_hP_bar.units = "m^3 s^-2"
    wcdf_hP_bar[:, :, :] = global_hP_bar[:, :, :]

    writeDS.close()

MPI.Finalize()
sys.exit()
