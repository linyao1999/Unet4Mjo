import numpy as np
import netCDF4 as nc4

def savenc(x,filename):
    f = nc4.Dataset(filename,'w', format='NETCDF4')
    f.createDimension('level', 2)
    f.createDimension('time', x.shape[0])
    
    psi = f.createVariable('PSI', 'f4', ('time','level'))
    psi[:,:] = x[:,:]
  
    f.close()

