import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
from functions import *


def writeAdditionalVariables(readFolder, writeFolder, curDateTime, gridFile):

    xds = xr.open_dataset(gridFile)
    dx = xds['dx'].to_numpy()
    dy = xds['dy'].to_numpy()
    xds.close()


    fname = f'GLORYS12v1_dailyAvg_{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d}.nc'
    wfname = f'GLORYS12v1_dailyAvg_{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d}_added.nc'
    
    if not os.path.isfile(readFolder+fname):
        print(fname, ' not present')
        return 
    xds = xr.open_dataset(readFolder + fname)

    uo = xds['uo']
    vo = xds['vo']
    thetao = xds['thetao']
    so = xds['so']
    zos = xds['zos']

    mask = np.isnan(uo.to_numpy())
    mask = np.logical_or(mask, np.isnan(vo.to_numpy() ))
    mask = np.logical_or(mask, np.isnan(thetao.to_numpy() ))
    mask = np.logical_or(mask, np.isnan(so.to_numpy() ))

    mask = np.logical_or(mask, abs(uo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(vo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(so.to_numpy())> 100)
    mask = np.logical_or(mask, abs(thetao.to_numpy())> 100)

    uo = xr.where(mask, np.nan, uo)
    vo = xr.where(mask, np.nan, vo)
    so = xr.where(mask, np.nan, so)
    thetao = xr.where(mask, np.nan, thetao)
    zos = xr.where(mask[:,0,:,:], np.nan, zos)


    uo = uo.interpolate_na(dim='latitude', method='nearest', limit=1)
    vo = vo.interpolate_na(dim='longitude', method='nearest', limit=1)

    vo = vo.interpolate_na(dim='latitude', method='nearest', limit=1)
    uo = uo.interpolate_na(dim='longitude', method='nearest', limit=1)

    so = so.interpolate_na(dim='latitude', method='nearest', limit=1)
    so = so.interpolate_na(dim='longitude', method='nearest', limit=1)

    thetao = thetao.interpolate_na(dim='latitude', method='nearest', limit=1)
    thetao = thetao.interpolate_na(dim='longitude', method='nearest', limit=1)

    latA = xds['latitude'].to_numpy()
    longA = xds['longitude'].to_numpy()
    depthA = xds['depth'].to_numpy()

    lat3DA = np.array(len(longA) *[latA], dtype=np.float64).T
    #print(lat3DA.shape)
    lat3DA = np.array([len(depthA) * [lat3DA]], dtype=np.float64)
    #print(lat3DA.shape)

    lon3DA = np.array(len(latA) *[longA], dtype=np.float64)
    lon3DA = np.array([len(depthA) * [lon3DA]], dtype=np.float64)

    omega = 7.2921159/1e5
    f = 2*omega*np.sin(np.deg2rad(lat3DA))

    #print(f.shape, depthA.shape, len(depthA))

    rhoA = get_density(so.to_numpy(), thetao.to_numpy(), depthA)
    presA = get_pressure(rhoA, zos.to_numpy(), depthA)
    woA = get_vertical_vel(uo.to_numpy(), vo.to_numpy(), dx, dy, depthA, mask)

    dx_presA, dy_presA = gethorizontal_grad(dx, dy, presA)
    
    dx_rhoA, dy_rhoA = gethorizontal_grad(dx, dy, rhoA)
    dz_rhoA = get_vertical_gradient(rhoA, depthA, mask, isscalar=True)

    dx_thetaoA, dy_thetaoA = gethorizontal_grad(dx, dy, thetao.to_numpy())
    dz_thetaoA = get_vertical_gradient(thetao.to_numpy(),depthA, mask, isscalar=True)

    dx_uoA, dy_uoA = gethorizontal_grad(dx, dy, uo.to_numpy())
    dz_uoA = get_vertical_gradient(uo.to_numpy(), depthA, mask )

    dx_voA, dy_voA = gethorizontal_grad(dx, dy, vo.to_numpy())
    dz_voA = get_vertical_gradient(vo.to_numpy(), depthA, mask )

    dx_woA, dy_woA = gethorizontal_grad(dx, dy, woA)
    dz_woA = get_vertical_gradient(woA, depthA, mask )

    rhoA[mask] = np.nan
    presA[mask] = np.nan
    woA[mask] = np.nan

    dx_presA[mask] = np.nan
    dy_presA[mask] = np.nan
    dx_rhoA[mask] = np.nan
    dy_rhoA[mask] = np.nan
    dz_rhoA[mask] = np.nan

    dx_thetaoA[mask] = np.nan
    dy_thetaoA[mask] = np.nan
    dz_thetaoA[mask] = np.nan

    dx_uoA[mask] = np.nan
    dy_uoA[mask] = np.nan
    dz_uoA[mask] = np.nan

    dx_voA[mask] = np.nan
    dy_voA[mask] = np.nan
    dz_voA[mask] = np.nan

    dx_woA[mask] = np.nan
    dy_woA[mask] = np.nan
    dz_woA[mask] = np.nan

    dimsList = ('time', 'depth', 'latitude', 'longitude')

    newVarsXds = xr.Dataset({
        'f':xr.DataArray(f, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'coriolis frequency'}),
        'rho':xr.DataArray(rhoA, dims=dimsList,   attrs= {'units':  'kg/m^3', 'long_name': 'density obtained using EOS'}),
        'pres':xr.DataArray(presA, dims=dimsList,   attrs= {'units':  'Pascal', 'long_name': 'from ssh and density'}),
        'wo':xr.DataArray(woA, dims=dimsList,   attrs= {'units':  'm/s', 'long_name':'vertical velcity obtained from div. U' }),
        'dx_pres':xr.DataArray(dx_presA, dims=dimsList,   attrs= {'units':  'Pascal/m', 'long_name': 'dx_Pres'}),
        'dy_pres':xr.DataArray(dy_presA, dims=dimsList,   attrs= {'units':  'Pascal/m', 'long_name': 'dy_Pres'}),
        'dx_rho':xr.DataArray(dx_rhoA, dims=dimsList,   attrs= {'units':  'kg/m^4', 'long_name': 'dx_rho'}),
        'dy_rho':xr.DataArray(dy_rhoA, dims=dimsList,   attrs= {'units':  'kg/m^4', 'long_name': 'dy_rho'}),
        'dz_rho':xr.DataArray(dz_rhoA, dims=dimsList,   attrs= {'units':  'kg/m^4', 'long_name': 'dz_rho'}),
        'dx_thetao':xr.DataArray(dx_thetaoA, dims=dimsList,   attrs= {'units': 'degC/m', 'long_name': 'dx of potential temperature'}),
        'dy_thetao':xr.DataArray(dy_thetaoA, dims=dimsList,   attrs= {'units': 'degC/m', 'long_name': 'dy of potential temperature'}),
        'dz_thetao':xr.DataArray(dz_thetaoA, dims=dimsList,   attrs= {'units': 'degC/m', 'long_name': 'dz of potential temperature'}),
        'dx_uo':xr.DataArray(dx_uoA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dx of u velocity'}),
        'dy_uo':xr.DataArray(dy_uoA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dy of u velocity'}),
        'dz_uo':xr.DataArray(dz_uoA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dz of u velocity'}),
        'dx_vo':xr.DataArray(dx_voA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dx of v velocity'}),
        'dy_vo':xr.DataArray(dy_voA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dy of v velocity'}),
        'dz_vo':xr.DataArray(dz_voA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dz of v velocity'}),
        'dx_wo':xr.DataArray(dx_woA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dx of w velocity'}),
        'dy_wo':xr.DataArray(dy_woA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dy of w velocity'}),
        'dz_wo':xr.DataArray(dz_woA, dims=dimsList,   attrs= {'units':  'sec^-1', 'long_name': 'dz of w velocity'}),
    })

    wxds = xr.merge([xds, newVarsXds])
    wxds.to_netcdf(writeFolder + wfname, unlimited_dims='time')

    wxds.close()
    xds.close()
    newVarsXds.close()



def main():
    readFolder = '/srv/cdx/hseo/Data/GLORYS12v1/TEP/2020/'
    writeFolder = '/srv/data1/particleTrack_GLORYS/partTracking/gloRYS_partTrack/prepareData/DATA/'
    gridFile = '/srv/data1/particleTrack_GLORYS/partTracking/gloRYS_partTrack/prepareData/glorysGrid_TEP.nc'
    startDate = datetime(2020,3,1)
    endDate = datetime(2020,6,30)

    curDateTime = startDate

    while curDateTime <= endDate:
        writeAdditionalVariables(readFolder, writeFolder, curDateTime, gridFile)
        curDateTime += timedelta(days=1)


if __name__ == "__main__":
    main()



