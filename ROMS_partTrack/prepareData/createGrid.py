from netCDF4 import Dataset
import numpy as np
import xarray as xr

fldLoc = './'
earthRad = 6.371e6
fileName = '/srv/seolab/srai/WPE/Run/old2/ocean_his.nc'
gridFileName = fldLoc + 'romsGrid_WEP.nc'
latVar = 'lat_rho'
lonVar = 'lon_rho'

ds = Dataset(fileName)

lat = np.array(ds.variables[latVar])
lon = np.array(ds.variables[lonVar])

eta_rho = np.arange(599)
xi_rho = np.arange(2599)

ds.close()

xlen = len(lon)
ylen = len(lat)

dx = np.zeros((ylen, xlen), dtype=float)
dy = np.zeros((ylen, xlen), dtype=float)

if np.max(abs(lat)) > 2:
    latInDeg = lat.copy()
    lonInDeg = lon.copy()%360
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
else:
    latInDeg = np.rad2deg(lat)
    lonInDeg = np.rad2deg(lon)%360


dlon = np.roll(lon, -1, axis=1) - lon
dlon[:,-1] = dlon[:,-2]
dlat = np.roll(lat, -1, axis=0) - lat
dlat[-1,:] = dlat[-2,:]

R = earthRad * np.cos(lat)
dx = R*dlon
dy = earthRad * dlat

coords = {'eta_rho': eta_rho, 'xi_rho': xi_rho}

# Create sample temperature and salinity data
# Create xarray dataset
xds = xr.Dataset(
    {
        'lat': (['eta_rho', 'xi_rho'], latInDeg, {'units': 'degrees N'}),
        'lon': (['eta_rho', 'xi_rho'], lonInDeg, {'units': 'degrees E'}),
        'dx': (['eta_rho', 'xi_rho'], dx, {'units': 'm'}),
        'dy': (['eta_rho', 'xi_rho'], dy, {'units': 'm'}),
        'area': (['eta_rho', 'xi_rho'], dx*dy, {'units': 'm^2'}),
    },
    coords=coords
)

xds.to_netcdf(gridFileName)

xds.close()




