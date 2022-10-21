import numpy as np 
import pandas as pd 
import xarray as xr 
import scipy.fft as fft 
import matplotlib.pyplot as plt 

# OLR
# remove 120-day running averages
varn = 'olr'
latsel = 90
latflg = 'GfltG'

# select the wanted time range and latitude range
ds0 = xr.open_dataset("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+".6hr.1978to2022.nc")
ds0 = ds0.sel(time=slice('1978-09-03', '2022-05-31'), lat=slice(latsel,-latsel))
ds0[varn] = - ds0[varn] / 3600   # convert the unit to W/m2
ds1 = ds0.sel(time=slice('1979-01-01', '2022-05-31'))

varn0 = ds0[varn]  # '1978-09-03', '2022-05-31'
varn1 = ds1[varn]  # '1979-01-01', '2022-05-31'

i = 0 
varnsel = varn0[i::4,:,:]  # select each day at 00:00/06:00/12:00/18:00
# print(varnsel)
fltu = varnsel.rolling(time=120, center=False).mean()
varnsel = varnsel.sel(time=slice('1979-01-01', '2022-05-31'))
varnsel = varnsel - fltu[119:-1,:,:].values

# print("removed 120-day running averages")
# print(varn)
# print(varnsel[0].time.dt.hour)
print("min: "+str(varnsel.min()))
print("max: " + str(varnsel.max()))
# print(ds1[varn][0,45,0])
# print(varnsel[0,45,0])
# print(fltu[119,45,0])
# print(varn1)