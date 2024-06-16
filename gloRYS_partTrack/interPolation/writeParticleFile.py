from netCDF4 import Dataset
import numpy as np
import xarray as xr

def createParticleFile(xpos, ypos, zpos, timeVal, varNameList, varUnitsList, varValList, fname):
    #xpos shape (time, pid)
    #varValList shape (nvars, time, pid)

    pids = np.arange(xpos.shape[1])
    xpos_darray = xr.DataArray(xpos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'degrees East Longitude', 
                                    'long_name': 'x position'})
    ypos_darray = xr.DataArray(ypos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'degrees North Latitude', 
                                    'long_name': 'x position'})
    zpos_darray = xr.DataArray(zpos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'meters', 
                                    'long_name': 'depth in meters'})
    
    dataset_dict = {}

    dataset_dict['xpos'] = xr.Dataset({'xpos': xpos_darray})
    dataset_dict['ypos'] = xr.Dataset({'ypos': ypos_darray})
    dataset_dict['zpos'] = xr.Dataset({'zpos': zpos_darray})

    for i in range(len(varNameList)):
        data_array = xr.DataArray(varValList[:,i,:], 
                                dims=('time', 'pid'),
                                coords={'time': timeVal, 'pid': pids}, 
                                attrs={'units': varUnitsList[i]})
        dataset_dict[varNameList[i]] = xr.Dataset({varNameList[i]: data_array})

    xds = xr.merge(list(dataset_dict.values()))
    xds.to_netcdf(fname, unlimited_dims='time')
    xds.close()

def appendParticleFile(xpos, ypos, zpos, timeVal, varNameList, varUnitsList, varValList, fname):
    #xpos shape (time, pid)
    #varValList shape (nvars, time, pid)
    xds = xr.open_dataset(fname)
    
    pids = np.arange(xpos.shape[1])

    xpos_darray = xr.DataArray(xpos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'degrees East Longitude', 
                                    'long_name': 'x position'})
    ypos_darray = xr.DataArray(ypos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'degrees North Latitude', 
                                    'long_name': 'x position'})
    zpos_darray = xr.DataArray(zpos[:,:], 
                            dims=('time', 'pid'),
                            coords={'time': timeVal, 'pid': pids}, 
                            attrs={'units':'meters', 
                                    'long_name': 'depth in meters'})
    
    dataset_dict = {}

    dataset_dict['xpos'] = xr.Dataset({'xpos': xpos_darray})
    dataset_dict['ypos'] = xr.Dataset({'ypos': ypos_darray})
    dataset_dict['zpos'] = xr.Dataset({'zpos': zpos_darray})

    for i in range(len(varNameList)):
        data_array = xr.DataArray(varValList[:,i,:], 
                                dims=('time', 'pid'),
                                coords={'time': timeVal, 'pid': pids}, 
                                attrs={'units': varUnitsList[i]})
        dataset_dict[varNameList[i]] = xr.Dataset({varNameList[i]: data_array})

    newxds = xr.merge(list(dataset_dict.values()))
    writexds = xr.concat([xds, newxds], dim='time')
    newxds.close()
    xds.close()
    
    writexds.to_netcdf(fname, unlimited_dims='time')
    writexds.close()