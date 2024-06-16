import numpy as np
from operators import *
import xarray as xr
import gc
import xroms

import matplotlib.pyplot as plt



def get_density3dDepth(salinity, pot_temp, depth):
    # ## the value of constants from paper https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015GL065525
    # g = 9.81 
    # alpha = 1.67e-4 #K^-1
    # gamma_b = 1.1179e-4 #(Km)^-1
    # gamma_c = 1e-5 #(K^-2)
    # beta = 0.78e-3 #(psu^-1)
    # c = 1500 #(m/s)  # this value from wikipedia

    # pot_temp0 = 10 # deg C
    # salinity0 = 35 # psu
    # rho0 = 1027 #kg/m^3

    # T1 = -g * depth/(c**2)
    # T2 = alpha*(1+gamma_b * depth)*(pot_temp - pot_temp0)
    # T3 = gamma_c/2 * (pot_temp - pot_temp0)**2
    # T4 = -beta *(salinity - salinity0)

    # b = g* (T1 + T2 + T3 + T4)
    # rho = -b* rho0/g + rho0

    R0 = 1027
    T0 = 0
    S0 = 35
    Tcoef = 1.7e-4
    Scoef = 7.6e-4

    rho = R0 - Tcoef*(pot_temp - T0) + Scoef*(salinity - S0)

    return rho

def get_pressure(Pair, density, z_w, z_rho):
    g = 9.81
    # tlen, zlen, ylen, xlen = z_rho.shape
    # p = np.zeros((tlen, zlen, ylen, xlen))
    # for i in range(zlen-1,-1,-1):
    #     dz = z_w[:,i+1,:,:] - z_rho[:,i,:,:]
    #     dp = density[:,i,:,:] * g * dz
    #     # plt.pcolormesh(z_rho[0,i,:,:])
    #     # plt.colorbar()
    #     # plt.show()
    #     if i == zlen-1:
    #         p[:,i,:,:] = dp + Pair
    #     else:
    #         p[:,i,:,:] = p[:,i+1,:,:] + dp
    p = -1027 * 9.81 * z_rho
    return p

def get_pressure_w(density, z_w):
    g = 9.81
    tlen, zlen, ylen, xlen = z_w.shape
    p = np.zeros((tlen, zlen, ylen, xlen))
    for i in range(zlen-1,-1,-1):
        if i == zlen-1:
            p[:,i,:,:] = 0
        else:
            dz = z_w[:,i+1,:,:] - z_w[:,i,:,:]
            dp = density[:,i,:,:] * g * dz
            p[:,i,:,:] = p[:,i+1,:,:] + dp
    return p


def createDataArray(varVal, dimsInTuple, coordsInDictWithVals, attrsInDict, dataType):
    darray = xr.DataArray(varVal[:,:], 
                            dims=dimsInTuple,
                            coords=coordsInDictWithVals,
                            dtype = dataType,
                            attrs=attrsInDict)
    return darray

def createDataset(varNameList, varValList, dimsList, coordsList, attrsList, dtypeList):
    dataset_dict = {}
    for i in range(len(varNameList)):
        data_array = xr.DataArray(varValList[i,:], 
                                dims=dimsList[i],
                                coords=coordsList[i], 
                                attrs=attrsList[i])
        
        dataset_dict[varNameList[i]] = xr.Dataset({varNameList[i]: data_array})

    newxds = xr.merge(list(dataset_dict.values()))

    return newxds


    
def writeAdditionalVariables_subDaily(inds, pm, pn, s_w, s_rho, f, writeFolder, curDateTime):
    print(curDateTime)
    wfname = f'ocean_his_{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d}_T{curDateTime.hour:02d}{curDateTime.minute:02d}{curDateTime.second:02d}_added.nc'
    xds = inds.sel(ocean_time=slice(f'{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d} {curDateTime.hour:02d}:{curDateTime.minute:02d}:{curDateTime.second:02d}',
                                  f'{curDateTime.year:04d}-{curDateTime.month:02d}-{curDateTime.day:02d} {curDateTime.hour:02d}:{curDateTime.minute:02d}:{curDateTime.second:02d}'))
    xds, xgrid = xroms.roms_dataset(xds)
    xds.xroms.set_grid(xgrid)

    z_w = xds['z_w']
    z_rho = xds['z_rho']

    uo = xds['u_eastward']
    vo = xds['v_northward']
    wo_w = xds['w']
    wo = 0.5*(wo_w.to_numpy()[:,0:-1,:,:] - wo_w.to_numpy()[:,1::,:,:])
    #Km = xds['AKv']
    
    Pair = xds['Pair']
    zos = xds['zeta']

    thetao = xds['temp']
    #Kt = xds['AKt']

    so = xds['salt']
    #Ks = xds['AKs']

    mask = np.isnan(uo.to_numpy())
    mask = np.logical_or(mask, np.isnan(vo.to_numpy() ))
    mask = np.logical_or(mask, abs(wo)> 100)
    #mask = np.logical_or(mask, np.isnan(Km.to_numpy() ))

    mask = np.logical_or(mask, np.isnan(thetao.to_numpy() ))
    #mask = np.logical_or(mask, np.isnan(Kt.to_numpy() ))

    mask = np.logical_or(mask, np.isnan(so.to_numpy() ))
    #mask = np.logical_or(mask, np.isnan(Ks.to_numpy() ))

    mask = np.logical_or(mask, np.isnan(so.to_numpy() ))
    mask = np.logical_or(mask, np.isnan(so.to_numpy() ))

    mask = np.logical_or(mask, abs(uo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(vo.to_numpy())> 100)
    mask = np.logical_or(mask, abs(wo)> 100)
    mask = np.logical_or(mask, abs(so.to_numpy())> 100)
    mask = np.logical_or(mask, abs(thetao.to_numpy())> 100)

    uo = xr.where(mask, np.nan, uo)
    vo = xr.where(mask, np.nan, vo)
    wo = xr.where(mask, np.nan, wo)

    so = xr.where(mask, np.nan, so)
    thetao = xr.where(mask, np.nan, thetao)

    zos = xr.where(mask[:,0,:,:], np.nan, zos)
    Pair = xr.where(mask[:,0,:,:], np.nan, Pair)

    t_len, zlen, ylen, xlen = uo.shape

    f_3DA = np.array([zlen * [f.to_numpy()]], dtype=np.float64)
    
    
    rhoA = get_density3dDepth(so.to_numpy(), thetao.to_numpy(), -z_rho.to_numpy())
    presA = get_pressure(Pair, rhoA, z_w.to_numpy(), z_rho.to_numpy())

    dxi_zA, deta_zA = gethorizontal_grad(pm, pn, z_rho.to_numpy())
    dsigma_zA = get_vertical_gradient_w(z_w.to_numpy(), s_w.to_numpy())   #### dz/dsigma
    # wo /= dsigma_zA  ### change back to pure omega

    dx_presA, dy_presA, dz_presA = calcAllThreeGradInCart(pm, pn, presA, s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    dx_thetaoA, dy_thetaoA, dz_thetaoA = calcAllThreeGradInCart(pm, pn, thetao.to_numpy(), s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    dx_soA, dy_soA, dz_soA = calcAllThreeGradInCart(pm, pn, so.to_numpy(), s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    dx_uoA, dy_uoA, dz_uoA = calcAllThreeGradInCart(pm, pn, uo.to_numpy(), s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    dx_voA, dy_voA, dz_voA = calcAllThreeGradInCart(pm, pn, vo.to_numpy(), s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    dx_woA, dy_woA, dz_woA = calcAllThreeGradInCart(pm, pn, wo, s_w.to_numpy(), dxi_zA, deta_zA, dsigma_zA)
    
    rhoA[mask] = np.nan
    presA[mask] = np.nan
    woA = wo#.to_numpy()

    dx_presA[mask] = np.nan
    dy_presA[mask] = np.nan
    dz_presA[mask] = np.nan
    
    dx_thetaoA[mask] = np.nan
    dy_thetaoA[mask] = np.nan
    dz_thetaoA[mask] = np.nan

    dx_soA[mask] = np.nan
    dy_soA[mask] = np.nan
    dz_soA[mask] = np.nan

    dx_uoA[mask] = np.nan
    dy_uoA[mask] = np.nan
    dz_uoA[mask] = np.nan

    dx_voA[mask] = np.nan
    dy_voA[mask] = np.nan
    dz_voA[mask] = np.nan

    dx_woA[mask] = np.nan
    dy_woA[mask] = np.nan
    dz_woA[mask] = np.nan

    dimsList = ('ocean_time', 's_rho', 'lat_rho', 'lon_rho')

    olvarsDS = xr.Dataset({'uo': uo, 
                            'vo': vo, 
                            'so': so, 
                            'thetao': thetao, 
                            'zos': zos,
                            'z_rho': z_rho})

    newVarsXds = xr.Dataset({
        'f':xr.DataArray(f_3DA, dims=dimsList,   attrs= {'units':  'sec^-1', 
                                                     'long_name': 'coriolis frequency'}),

        'rho':xr.DataArray(rhoA, dims=dimsList,   attrs= {'units':  'kg/m^3',
                                                         'long_name': 'density obtained using EOS'}),
        'pres':xr.DataArray(presA, dims=dimsList,   attrs= {'units':  'Pascal',
                                                            'long_name': 'from ssh and density'}),
        'wo':xr.DataArray(woA, dims=dimsList,   attrs= {'units':  'm/s',
                                                        'long_name':'vertical velcity omega at cell center calculated by averaging' }),

        'dxi_z':xr.DataArray(dxi_zA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dx_Pres'}),
        'deta_z':xr.DataArray(deta_zA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dy_Pres'}),
        'dsigma_z':xr.DataArray(dsigma_zA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dy_Pres'}),

        'dx_pres':xr.DataArray(dx_presA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dx_Pres'}),
        'dy_pres':xr.DataArray(dy_presA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dy_Pres'}),
        'dz_pres':xr.DataArray(dy_presA, dims=dimsList,   attrs= {'units':  'Pascal/m',
                                                                  'long_name': 'dy_Pres'}),
        'dx_thetao':xr.DataArray(dx_thetaoA, dims=dimsList,   attrs= {'units': 'degC/m',
                                                                      'long_name': 'dx of potential temperature'}),
        'dy_thetao':xr.DataArray(dy_thetaoA, dims=dimsList,   attrs= {'units': 'degC/m',
                                                                      'long_name': 'dy of potential temperature'}),
        'dz_thetao':xr.DataArray(dz_thetaoA, dims=dimsList,   attrs= {'units': 'degC/(s_cord_unit)',
                                                                      'long_name': 'dz of potential temperature'}),
        'dx_uo':xr.DataArray(dx_uoA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dx of u velocity'}),
        'dy_uo':xr.DataArray(dy_uoA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dy of u velocity'}),
        'dz_uo':xr.DataArray(dz_uoA, dims=dimsList,   attrs= {'units':  'm/sec^-1/(s_cord_unit)',
                                                              'long_name': 'dz of u velocity'}),
        'dx_vo':xr.DataArray(dx_voA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dx of v velocity'}),
        'dy_vo':xr.DataArray(dy_voA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dy of v velocity'}),
        'dz_vo':xr.DataArray(dz_voA, dims=dimsList,   attrs= {'units':  'm/sec^-1/(s_cord_unit)',
                                                              'long_name': 'dz of v velocity'}),
        'dx_wo':xr.DataArray(dx_woA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dx of w velocity'}),
        'dy_wo':xr.DataArray(dy_woA, dims=dimsList,   attrs= {'units':  'sec^-1',
                                                              'long_name': 'dy of w velocity'}),
        'dz_wo':xr.DataArray(dz_woA, dims=dimsList,   attrs= {'units':  'm/sec^-1/(s_cord_unit)',
                                                              'long_name': 'dz of w velocity'}),
    })

    del dx_presA
    del dy_presA
    del dz_presA
    del dx_thetaoA
    del dy_thetaoA
    del dz_thetaoA
    del dx_soA
    del dy_soA
    del dz_soA
    del dx_uoA
    del dy_uoA
    del dz_uoA
    del dx_voA
    del dy_voA
    del dz_voA
    del dx_woA
    del dy_woA
    del dz_woA
    
    wxds = xr.merge([olvarsDS, newVarsXds])
    wxds.to_netcdf(writeFolder + wfname, unlimited_dims='time')

    xds.close()
    wxds.close()
    newVarsXds.close()

    del xds, wxds, newVarsXds
    gc.collect() ### for freeing memory #garbage collector



