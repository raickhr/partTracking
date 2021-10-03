import numpy as np
from netCDF4 import Dataset
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
cur_xpos, cur_ypos = config.getInitialParticlePos()

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
        '_{0:03d}p_{1:s}_{2:02d}substeps_{3:s}.nc'.format(config.nParticles, suffix, config.nSubTimeStep, \
                                                          config.spaceInterpolation)

    timeVal = np.array(ds.variables['Time'])
    timeUnits = ds.variables['Time'].units
    timelen = len(timeVal)

    dt = (timeVal[1] - timeVal[0])  # *24*3600

    all_xpos = np.zeros((timelen, config.nParticles), dtype = float)
    all_ypos = np.zeros((timelen, config.nParticles), dtype = float)

    writeFieldsVals = np.zeros((config.nWriteFields, timelen, config.nParticles), dtype = float) 

    u = np.array(ds.variables[config.uVarName])
    v = np.array(ds.variables[config.vVarName])

    if fileNum > 0:
        prev_ds = Dataset(config.locationFolder + config.filteredFileNameList[fileNum -1])
        prev_timeLen = len(np.array(prev_ds.variables['Time']))
        t1 = prev_timeLen -2
        u_prev = np.array(prev_ds.variables[config.uVarName][t1:prev_timeLen, :, :])
        v_prev = np.array(prev_ds.variables[config.vVarName][t1:prev_timeLen, :, :])
        u = np.concatenate((u_prev, u), axis = 0)
        v = np.concatenate((v_prev, v), axis = 0)



    if fileNum < (nFiles -1):
        next_ds = Dataset(config.locationFolder + config.filteredFileNameList[fileNum + 1])
        u_next = np.array(next_ds.variables[config.uVarName][0:2, :, :])
        v_next = np.array(next_ds.variables[config.vVarName][0:2, :, :])
        u = np.concatenate((u, u_next), axis=0)
        v = np.concatenate((v, v_next), axis=0)


    for timeIndex in range(timelen):
        print('time count', timeIndex)
        shiftIndex = timeIndex + 2
        if fileNum == 0:
            shiftIndex = timeIndex
        ts = shiftIndex - 2
        te = shiftIndex + 2 
        writeNan = False
        if ts < 0 or ((fileNum >= nFiles-1) and (timeIndex >= timelen-2)):
            all_xpos[timeIndex, :] = float('nan')
            all_ypos[timeIndex, :] = float('nan')

            writeFieldsVals[:, timeIndex, :] = float('nan')

        else:
            xvel_inGrid = u[ts:te,:,:]
            yvel_inGrid = v[ts:te,:,:]

            #reading fields from netcdf file
            allFields = np.zeros((config.nWriteFields, ylen, xlen), dtype=float)

            for fieldCount in range(config.nWriteFields):
                varName = config.writeFieldsName[fieldCount]
                allFields[fieldCount,:,:] = np.array(ds.variables[varName][timeIndex,:,:])

            #interpolating to particle positions
            allInterPolatedFields = get_fields_at_loc(
                cur_xpos, cur_ypos, xh, yh, config.nWriteFields, allFields)

            all_xpos[timeIndex, :] = cur_xpos
            all_ypos[timeIndex, :] = cur_ypos

            writeFieldsVals[:, timeIndex, :] = allInterPolatedFields[:,:]

            new_xpos, new_ypos = [], []
            if config.timeIntegrationMethod == 'RK4': # Runga Kutta 4
                print('RK4 time integrating')
                new_sub_dt = dt/config.nSubTimeStep
                for sub_dt_count in range(config.nSubTimeStep):
                    
                    #RK4 requires fieldVals for 3 time locations at t_n, t_(n+1/2) and t_(n + 1)
                    xvel_in_0sub_dt = get_field_at_time(sub_dt_count * new_sub_dt, ylen, xvel_inGrid, dt)
                    yvel_in_0sub_dt = get_field_at_time(sub_dt_count * new_sub_dt, ylen, yvel_inGrid, dt)

                    xvel_in_half_sub_dt = get_field_at_time((sub_dt_count+0.5) * new_sub_dt, ylen, xvel_inGrid, dt)
                    yvel_in_half_sub_dt = get_field_at_time((sub_dt_count+0.5) * new_sub_dt, ylen, yvel_inGrid, dt)

                    xvel_in_1sub_dt = get_field_at_time((sub_dt_count+1) * new_sub_dt, ylen, xvel_inGrid, dt)
                    yvel_in_1sub_dt = get_field_at_time((sub_dt_count+1) * new_sub_dt, ylen, yvel_inGrid, dt)


                    xvel_inGrid_3 = np.stack((xvel_in_0sub_dt,xvel_in_half_sub_dt,xvel_in_1sub_dt), axis = 0)
                    yvel_inGrid_3 = np.stack((yvel_in_0sub_dt,yvel_in_half_sub_dt,yvel_in_1sub_dt), axis = 0)
                    
                    new_xpos, new_ypos = updatePositions_RK4(cur_xpos, cur_ypos, 
                                                            xvel_inGrid_3, yvel_inGrid_3,
                                                            new_sub_dt, xh, yh, intp=config.spaceInterpolation)

                    cur_xpos = new_xpos
                    cur_ypos = new_ypos

            elif config.timeIntegrationMethod == 'PCE':  # predictor corrector method
                print('PCE time integrating')
                new_sub_dt = dt/config.nSubTimeStep
                if config.nSubTimeStep == 1:
                    cur_xvel_inGrid = xvel_inGrid[1, :,:]
                    next_xvel_inGrid = xvel_inGrid[2, :, :]

                    cur_yvel_inGrid = yvel_inGrid[1, :, :]
                    next_yvel_inGrid = yvel_inGrid[2, :, :]

                    #print(np.shape(cur_xvel_inGrid))
                    new_xpos, new_ypos = updatePositions_predictorCorrector(cur_xpos, cur_ypos,
                                                                            cur_xvel_inGrid, cur_yvel_inGrid,
                                                                            next_xvel_inGrid, next_yvel_inGrid,
                                                                            new_sub_dt, xh, yh, intp=config.spaceInterpolation)
                    cur_xpos = new_xpos
                    cur_ypos = new_ypos
                
                else:
                    for sub_dt_count in range(config.nSubTimeStep):
                        cur_xvel_inGrid = get_field_at_time(sub_dt_count * new_sub_dt, ylen, xvel_inGrid, dt)
                        next_xvel_inGrid = get_field_at_time((sub_dt_count +1)* new_sub_dt, ylen, xvel_inGrid, dt)
                        
                        cur_yvel_inGrid = get_field_at_time(sub_dt_count * new_sub_dt, ylen, yvel_inGrid, dt)
                        next_yvel_inGrid = get_field_at_time((sub_dt_count +1)* new_sub_dt, ylen, yvel_inGrid, dt)
                        
                        #print(np.shape(cur_xvel_inGrid))
                        new_xpos, new_ypos = updatePositions_predictorCorrector(cur_xpos, cur_ypos,
                                                                            cur_xvel_inGrid, cur_yvel_inGrid,
                                                                            next_xvel_inGrid, next_yvel_inGrid,
                                                                                new_sub_dt, xh, yh, intp=config.spaceInterpolation)
                        cur_xpos = new_xpos
                        cur_ypos = new_ypos


    
    writeDS = Dataset(wFname, 'w', format='NETCDF4_CLASSIC')
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
    
