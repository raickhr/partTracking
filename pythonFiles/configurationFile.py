import numpy as np
import csv
from netCDF4 import Dataset
import json

class configClass:
    def __init__(self, fileName):
        # Opening JSON file
        f = open(fileName)
        data = json.load(f)

        #data location Folder
        self.locationFolder = data["locationFolder"]

        #boolean to see if initial Particle positions are available
        self.initialParticlePosition = data["initialParticlePosition"]

        #name of initial particle position file
        self.initialParticlePositionFile = data["initialParticlePositionFile"]

        #number of particles
        self.nParticles = data["nParticles"]
        
        #list of filenames of filtered fields
        self.filteredFileNameList = data["filteredFileNameList"]

        self.timeIntegrationMethod = data["timeIntegrationMethod"]
        self.nSubTimeStep = data["nSubTimeStep"]
        self.spaceInterpolation = data["spaceInterpolation"]


        self.uVarName = data["uVarName"]
        self.vVarName = data["vVarName"]

        #fields to be written in output
        self.nWriteFields = data["nWriteFields"]
        self.writeFieldsName = data["writeFieldsName"]
        
        f.close()

    def getInitialParticlePos(self ):
        xpos = []
        ypos = []
        if self.initialParticlePosition:
            file = open(self.locationFolder + self.initialParticlePositionFile)
            csvreader = csv.reader(file)
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
            file.close()

            rows = np.array(rows, dtype=np.float64)

            xpos = rows[:, 0]
            ypos = rows[:, 1]

            self.nParticles = len(xpos)


        else:
            ds = Dataset(self.filteredFileNameList[0])
            xh = np.array(ds.variables['xh'])
            yh = np.array(ds.variables['yh'])
            xlen = len(xh)
            ylen = len(yh)
            xpos = np.random.rand(self.nParticles) * (xh[xlen-1] - xh[0]) + xh[0]
            ypos = np.random.rand(self.nParticles) * (yh[ylen-1] - yh[0]) + yh[0]

            data = np.stack((xpos, ypos), axis=1)
            header = ['xpos', 'ypos']
            with open(self.initialParticlePositionFile, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # 2. create a csvwriter object
                csvwriter.writerow(header)  # 4. write the header
                csvwriter.writerows(data)  # 5. write the rest of the data

        return xpos, ypos

class configClass2:
    def __init__(self, fileName):
        # Opening JSON file
        f = open(fileName)
        data = json.load(f)

        #particle file location Folder
        self.particleFileLocationFolder = data["particleFileLocationFolder"]

        #filtered file location Folder
        self.filteredFileLocationFolder = data["filteredFileLocationFolder"]

        #particle file suffix
        self.particleFileSuffix = data["particleFileSuffix"]

        #filtered file suffix
        self.filteredFileSuffix = data["filteredFileSuffix"]

        #particle file suffix
        self.particleFilePrefix = data["particleFilePrefix"]

        #filtered file suffix
        self.filteredFilePrefix = data["filteredFilePrefix"]

        #particle file suffix
        self.fileStartCount = data["fileStartCount"]

        #filtered file suffix
        self.fileEndCount = data["fileEndCount"]

        #number of particles
        self.nParticles = data["nParticles"]

        #write folder location
        self.writeLocation = data["writeLocation"]

        #write file prefix
        self.writeFilePrefix = data["writeFilePrefix"]

        #write file suffix
        self.writeFileSuffix = data["writeFileSuffix"]

        #fields to be written in output
        self.nWriteFields = data["nWriteFields"]
        self.writeFieldsName = data["writeFieldsName"]

        f.close()

    def getnFiles(self):
        return self.fileEndCount - self.fileStartCount + 1

    def getParticleFileName(self, fileCount):
        #fileCount starts from zero
        fileNumber = self.fileStartCount + fileCount
        fileName = self.particleFileLocationFolder + '/' + self.particleFilePrefix + \
            '{0:03d}'.format(fileNumber) + self.particleFileSuffix
        return fileName

    def getFilteredFileName(self, fileCount):
        #fileCount starts from zero
        fileNumber = self.fileStartCount + fileCount
        fileName = self.filteredFileLocationFolder + '/' + self.filteredFilePrefix + \
            '{0:03d}'.format(fileNumber) + self.filteredFileSuffix
        return fileName


    def makeWriteFileName(self, fileCount):
        #fileCount starts from zero
        fileNumber = self.fileStartCount + fileCount
        fileName = self.writeLocation + '/' + self.writeFilePrefix + \
            '{0:03d}'.format(fileNumber) + self.writeFileSuffix
        return fileName
