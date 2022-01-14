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

config = configClass2(configFile)

nFiles = config.getnFiles()

firstFile = config.getFilteredFileName(0)

ds = Dataset(firstFile)
xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])

dx = xh[1] - xh[0]
dy = yh[1] - yh[0]

xlen = len(xh)
ylen = len(yh)

for fileNum in range(nFiles):
    print('fileCount', fileNum ) 
    

    filteredFile = config.getFilteredFileName(fileNum)
    ds = Dataset(filteredFile)
    
    particleFile = config.getParticleFileName(fileNum)
    ds2 = Dataset(particleFile)

    writeFile = config.makeWriteFileName(fileNum)

    timeVal = np.array(ds.variables['Time'])
    timeUnits = ds.variables['Time'].units
    timelen = len(timeVal)

    all_xpos = np.array(ds2.variables['xpos'][:,:], dtype= float)
    all_ypos = np.array(ds2.variables['ypos'][:, :], dtype=np.float64)

    checkNtimeLen, checkNparticles = np.shape(all_xpos)

    if config.nParticles != checkNparticles or timelen != checkNtimeLen:
        print('error in size of particles or time')
        sys.exit()

    writeFieldsVals = np.zeros(
        (config.nWriteFields, timelen, config.nParticles), dtype=np.float64)

    for timeIndex in range(timelen):
            #print('time count', timeIndex)
            cur_xpos = all_xpos[timeIndex, :]
            cur_ypos = all_ypos[timeIndex, :]

            #reading fields from netcdf file
            allFields = np.zeros(
                (config.nWriteFields, ylen, xlen), dtype=np.float64)

            for fieldCount in range(config.nWriteFields):
                varName = config.writeFieldsName[fieldCount]
                allFields[fieldCount, :, :] = np.array(
                    ds.variables[varName][timeIndex, :, :])

            #interpolating to particle positions
            allInterPolatedFields = get_fields_at_loc(
                cur_xpos, cur_ypos, xh, yh, config.nWriteFields, allFields)


            writeFieldsVals[:, timeIndex, :] = allInterPolatedFields[:, :]

            
    writeDS = Dataset(writeFile, 'w', format='NETCDF4_CLASSIC')
    writeDS.createDimension('Time', None)
    writeDS.createDimension('PID', config.nParticles)

    wCDF_Time = writeDS.createVariable('Time', np.float64, ('Time'))
    wCDF_Time.units = timeUnits
    wCDF_Time[0:timelen] = timeVal[0:timelen]

    wCDF_xpos = writeDS.createVariable('xpos', np.float64, ('Time', 'PID'))
    wCDF_ypos = writeDS.createVariable('ypos', np.float64, ('Time', 'PID'))
    wCDF_xpos[:, :] = all_xpos[:, :]
    wCDF_ypos[:, :] = all_ypos[:, :]

    for fieldCount in range(config.nWriteFields):
        varName = config.writeFieldsName[fieldCount]
        wCDF_var = writeDS.createVariable(varName, np.float64, ('Time', 'PID'))
        wCDF_var[:, :] = writeFieldsVals[fieldCount, :, :]

    writeDS.close()
