import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta
import os
from functions import *

def main():
    readFolder = '/srv/seolab/srai/WPE/Run/' #'/srv/data1/particleTrack_GLORYS/partTracking/ROMS_partTrack/prepareData/DATA_subDaily/'
    writeFolder = '/srv/data1/particleTrack_GLORYS/partTracking/ROMS_partTrack/prepareData/DATA_subDaily/'
    gridFile = '/srv/data1/particleTrack_GLORYS/partTracking/ROMS_partTrack/prepareData/DATA/romsGrid_WEP.nc'
    startDate = datetime(2018,3,12)
    endDate = datetime(2018,4,2)

    curDateTime = startDate

    fname = readFolder + 'ocean_his.nc'
    if not os.path.isfile(fname):
        print(fname, ' not present')
        return 
    
    ds = xr.open_dataset(fname)
    lat_rho = ds['lat_rho']
    lon_rho = ds['lon_rho']
    s_rho = ds['s_rho']
    s_w = ds['s_w']
    pm  = ds['pm'].to_numpy()
    pn = ds['pn'].to_numpy()
    f = ds['f']
    
    while curDateTime <= endDate:
        writeAdditionalVariables_subDaily(ds, pm, pn, s_w, s_rho, f, writeFolder, curDateTime)
        curDateTime += timedelta(minutes=60)
        


if __name__ == "__main__":
    main()



