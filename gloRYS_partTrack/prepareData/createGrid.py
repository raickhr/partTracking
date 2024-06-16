from netCDF4 import Dataset
import numpy as np
import xarray as xr

fldLoc = './'
earthRad = 6.371e6
fileName = '/srv/cdx/hseo/Data/GLORYS12v1/TEP/2020/GLORYS12v1_dailyAvg_2020-01-01.nc'
gridFileName = fldLoc + 'glorysGrid_TEP.nc'
latVar = 'latitude'
lonVar = 'longitude'

ds = Dataset(fileName)

lat = np.array(ds.variables[latVar])
lon = np.array(ds.variables[lonVar])

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


dlon = abs(lon[1] - lon[0])
dlat = abs(lat[1] - lat[0])

for i in range(ylen):
    R = earthRad * np.cos(lat[i])
    dx[i,:] = R*dlon
    dy[i,:] = earthRad * dlat

coords = {'lat': latInDeg, 'lon': lonInDeg}

# Create sample temperature and salinity data
# Create xarray dataset
xds = xr.Dataset(
    {
        'dx': (['lat', 'lon'], dx, {'units': 'm'}),
        'dy': (['lat', 'lon'], dy, {'units': 'm'}),
        'area': (['lat', 'lon'], dx*dy, {'units': 'm^2'}),
    },
    coords=coords
)

xds.to_netcdf(gridFileName)

xds.close()




