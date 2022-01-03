from netCDF4 import Dataset
import numpy as np
from configurationFile import *
from lagrangianParticleFuncs import *
import argparse
import sys


parser = argparse.ArgumentParser()

parser.add_argument("--file", "-f", type=str, action='store', required=True,
                    help="configuration file ")

args = parser.parse_args()

configFile = args.file

config = configClass(configFile)

nFiles = len(config.filteredFileNameList)

ds = Dataset(config.locationFolder + config.filteredFileNameList[0])
xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])

dx = xh[1] - xh[0]
dy = yh[1] - yh[0]

xlen = len(xh)
ylen = len(yh)

for fileNum in range(nFiles):
    print('fileCount', fileNum, config.filteredFileNameList[fileNum])
    ds = Dataset(config.locationFolder + config.filteredFileNameList[fileNum])
    suffix = ''
    if config.timeIntegrationMethod == 'RK4':
        suffix = 'RK4'
    elif config.timeIntegrationMethod == 'PCE':
        suffix = 'PCE'

    wFname = config.locationFolder + '/' + config.filteredFileNameList[fileNum].rstrip('.nc') + \
        '_{0:03d}p_{1:s}_{2:02d}substeps_{3:s}.nc'.format(config.nParticles, suffix, config.nSubTimeStep,
                                                          config.spaceInterpolation)

    stripText = '_{0:03d}p_{1:s}_{2:02d}substeps_{3:s}.nc'.format(config.nParticles, suffix, config.nSubTimeStep,
                                                                       config.spaceInterpolation)

    ds2 = Dataset(wFname)
    
    wFname_2 = wFname.rstrip(stripText) + '_pTrack.nc'    

    timeVal = np.array(ds.variables['Time'])
    timeUnits = ds.variables['Time'].units
    timelen = len(timeVal)

    all_xpos = np.array(ds2.variables['xpos'][:,:], dtype= float)
    all_ypos = np.array(ds2.variables['ypos'][:, :], dtype=float)

    writeFieldsVals = np.zeros(
        (config.nWriteFields, timelen, config.nParticles), dtype=float)

    for timeIndex in range(timelen):
            print('time count', timeIndex)
            cur_xpos = all_xpos[timeIndex, :]
            cur_ypos = all_ypos[timeIndex, :]

            #reading fields from netcdf file
            allFields = np.zeros(
                (config.nWriteFields, ylen, xlen), dtype=float)

            for fieldCount in range(config.nWriteFields):
                varName = config.writeFieldsName[fieldCount]
                allFields[fieldCount, :, :] = np.array(
                    ds.variables[varName][timeIndex, :, :])

            #interpolating to particle positions
            allInterPolatedFields = get_fields_at_loc(
                cur_xpos, cur_ypos, xh, yh, config.nWriteFields, allFields)


            writeFieldsVals[:, timeIndex, :] = allInterPolatedFields[:, :]

            
    writeDS = Dataset(wFname_2, 'w', format='NETCDF4_CLASSIC')
    writeDS.createDimension('Time', None)
    writeDS.createDimension('PID', config.nParticles)

    wCDF_Time = writeDS.createVariable('Time', np.float32, ('Time'))
    wCDF_Time.units = timeUnits
    wCDF_Time[0:timelen] = timeVal[0:timelen]

    wCDF_xpos = writeDS.createVariable('xpos', np.float32, ('Time', 'PID'))
    wCDF_ypos = writeDS.createVariable('ypos', np.float32, ('Time', 'PID'))
    wCDF_xpos[:, :] = all_xpos[:, :]
    wCDF_ypos[:, :] = all_ypos[:, :]

    for fieldCount in range(config.nWriteFields):
        varName = config.writeFieldsName[fieldCount]
        wCDF_var = writeDS.createVariable(varName, np.float32, ('Time', 'PID'))
        wCDF_var[:, :] = writeFieldsVals[fieldCount, :, :]

    writeDS.close()
