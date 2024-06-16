import numpy as np
from operators import *
import xarray as xr


def get_density(salinity, pot_temp, depth1d):
    ## the value of constants from paper https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015GL065525
    g = 9.81 
    alpha = 1.67e-4 #K^-1
    gamma_b = 1.1179e-4 #(Km)^-1
    gamma_c = 1e-5 #(K^-2)
    beta = 0.78e-3 #(psu^-1)
    c = 1500 #(m/s)  # this value from wikipedia

    depth = np.zeros(salinity.shape, dtype=np.float64)
    for i in range(len(depth1d)):
        depth[:,i,:,:] = depth1d[i]

    pot_temp0 = 10 # deg C
    salinity0 = 35 # psu
    rho0 = 1027 #kg/m^3

    T1 = -g * depth/(c**2)
    T2 = alpha*(1+gamma_b * depth)*(pot_temp - pot_temp0)
    T3 = gamma_c/2 * (pot_temp - pot_temp0)**2
    T4 = -beta *(salinity - salinity0)

    b = g* (T1 + T2 + T3 + T4)
    rho = -b* rho0/g + rho0

    return rho

def get_pressure(density, ssh, depth):
    g = 9.81
    p = np.zeros(density.shape, dtype=np.float64)
    for k in range(len(depth)):
        if k == 0:
            intfDepth = 0
            upperPres = density[:,0,:,:] * g * ssh[:,:,:]
        else:
            intfDepth = (depth[k]  + depth[k-1])/2
            upperPres = p[:,k-1,:,:]

        dz = depth[k] - intfDepth
        p[:,k,:,:] = upperPres + density[:,k,:,:] * g * dz

    return p
            

def get_vertical_vel(u, v, dx, dy, depth, mask):
    depthLabel = np.sum(mask[0,:,:,:], axis=0)-2
    horiz_div = gethorizontal_div(dx, dy, u, v, mask)
    horiz_div[mask] = 0.0
    w = np.zeros(horiz_div.shape, dtype=np.float64)
    for k in range(len(depth)-2,-1,-1):
        mmask = depthLabel < k
        dz = (depth[k+1] - depth[k])  ### This is actually -dz
        wlower = w[:,k+1,:,:]
        horiz_div[:,k,mmask] = 0
        w[:,k,:,:] = dz*horiz_div[:,k,:,:] + wlower
        w[:,k,mmask] = 0  
    return w

def get_vertical_gradient(phi, depth, mask, isscalar=False):
    depthLabel = np.sum(mask[0,:,:,:], axis=0)-2
    grad = np.zeros(phi.shape, dtype=np.float64)
    if isscalar:
        phi = xr.DataArray(phi, dims=['time','depth','latitude','longitude'])
        phi = phi.ffill(dim="depth", limit=None)
        phi = phi.to_numpy()
    for k in range(len(depth)-2):
        mmask = depthLabel < k
        dz = (depth[k+1] - depth[k])  ### This is actually -dz
        dphi = phi[:,k,:,:] - phi[:,k+1,:,:]
        #print(dphi.shape, dz)
        #print(type(dphi), type(dz))
        grad[:,k,:,:] = dphi/dz
        if not isscalar:
            grad[:,k,mmask] = 0
    
    return grad



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



