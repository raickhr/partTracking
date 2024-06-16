from filteringFunctions import *
from netCDF4 import Dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--inputFile", "-f", type=str, default='prog_RequiredFieldsOnly_4FilteredFields.nc', action='store',
                    help="this is the output file prom MOM6")

parser.add_argument("--fldLoc", "-l", type=str, default='.', action='store',
                    help="this is the location of the output file from MOM6")


args = parser.parse_args()

fileName = args.inputFile
fldLoc = args.fldLoc

readFileName = fldLoc + '/' + fileName

ds= Dataset(readFileName)

f0 = 6.49e-5
beta = 2e-11

u = np.array(ds.variables['u'])
v = np.array(ds.variables['v'])
xh = np.array(ds.variables['xh'])
yh = np.array(ds.variables['yh'])
yCenter = np.mean(yh)

f = f0 + beta * yh
