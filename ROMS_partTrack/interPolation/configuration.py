import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
import os
import csv
import json

class configClass:
    def __init__(self):
        self.startdate = datetime(2018,3,12)
        self.enddate = datetime(2018,3,20)

        self.varNameList = ['uo',  'vo',  'wo', 'z_rho', 'dxi_z', 'deta_z' , 'dsigma_z', 'dz_uo', 'dz_vo', 'dz_wo', 'rho',    'pres',   'dx_pres',  'dy_pres',  'thetao', 'so', 'f']
        self.varUnitsList = ['m/s','m/s', 'm/s','m',      '',     '',        'm',        'm/s',   'm/s',   'm/s',   'kg/m^3', 'Pascal', 'Pascal/m', 'Pascal/m', 'deg C',  'psu', 'sec^-1']

        self.timeVarName = 'ocean_time'
        self.xcoord = 'lon_rho'
        self.ycoord = 'lat_rho'
        self.zcoord = 's_rho'

        self.writeFileName = 'particleTracks.nc'
        self.timeDiffConscFiles = timedelta(minutes=60) #timedelta(days=1)
        self.nsubTimeSteps = 1

        self.readFolder = '/srv/data1/particleTrack_GLORYS/partTracking/ROMS_partTrack/prepareData/DATA_subDaily/'
        self.writeFolder = '/srv/data1/particleTrack_GLORYS/partTracking/ROMS_partTrack/subDailyOutput/'
        self.startFile = f'ocean_his_{self.startdate.year:04d}-{self.startdate.month:02d}-{self.startdate.day:02d}_T{self.startdate.hour:02d}{self.startdate.minute:02d}{self.startdate.second:02d}_added.nc'

        self.initPartPosFile = 'initPos_0.csv'
        self.gridFile = 'glorysGrid_TEP.nc'

        self.xvelVarIndx = 0
        self.yvelVarIndx = 1
        self.zvelVarIndx = 2
        self.cartZvarIndx = 3
        self.xSigmaSlopeIndx = 4
        self.ySigmaSlopeIndx = 5
        self.thicknessPerSigmaIndx = 6

        self.manualInitPos = True
        self.nParticles = 1
        
    def getStartPartPos(self):
        xpos = []
        ypos = []
        zpos = []

        if self.manualInitPos:
            file = open(self.writeFolder + self.initPartPosFile)
            csvreader = csv.reader(file, skipinitialspace=True, delimiter=',')
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append([x.strip() for x in row])
            file.close()
            #print(rows)
            rows = np.array(rows, dtype=np.float64)

            xpos = rows[:, 0]
            ypos = rows[:, 1]
            zpos = rows[:, 2]

            self.nParticles = len(xpos)


        else:
            ds = Dataset(self.readFolder + self.startFile)
            xh = np.array(ds.variables[self.xcoord])
            yh = np.array(ds.variables[self.ycoord])
            zh = np.array(ds.variables[self.zcoord])
            xlen = len(xh)
            ylen = len(yh)
            zlen = len(zh)
            ds.close()

            xpos = np.random.rand(self.nParticles) * (xh[xlen-1] - xh[0]) + xh[0]
            ypos = np.random.rand(self.nParticles) * (yh[ylen-1] - yh[0]) + yh[0]
            zpos = np.random.rand(self.nParticles) * (zh[zlen-1] - zh[0]) + zh[0]

            data = np.stack((xpos, ypos, zpos), axis=1)
            header = ['xpos', 'ypos', 'zpos']
            with open(self.writeFolder+ self.initPartPosFile, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # 2. create a csvwriter object
                csvwriter.writerow(header)  # 4. write the header
                csvwriter.writerows(data)  # 5. write the rest of the data

        return xpos, ypos, zpos
   
    def getCoordinates(self):
        ds = Dataset(self.readFolder + self.startFile)
        ycoords = np.array(ds.variables['lat_rho'][:,0])
        xcoords = np.array(ds.variables['lon_rho'][0,:])
        zcoords = np.array(ds.variables['s_rho'])
        ds.close()

        return xcoords, ycoords, zcoords
    


    def getTimeList(self):
        timeList = []
        curDate = self.startdate
        while curDate < self.enddate:
            timeList.append(curDate)
            curDate += self.timeDiffConscFiles
        
        return timeList
    
    def getFileName(self, time):
        fullFileName = f'ocean_his_{time.year:04d}-{time.month:02d}-{time.day:02d}_added.nc'
        if os.path.isfile(fullFileName):
            return fullFileName
        else:
            fullFileName = f'ocean_his_{time.year:04d}-{time.month:02d}-{time.day:02d}_T{time.hour:02d}{time.minute:02d}{time.second:02d}_added.nc' 
            return fullFileName

        

    def getTimeVal(self, file):
        ds = Dataset(file)
        time = np.array(ds.variables[self.timeVarName])
        units = ds.variables[self.timeVarName].units
        cdfDate = num2date(time[0], units)
        datetimeVal = datetime(cdfDate.year, 
                        cdfDate.month,
                        cdfDate.day,
                        cdfDate.hour,
                        cdfDate.minute,
                        cdfDate.second)
        
        ds.close()
        
        return datetimeVal








    
