# calculate anomalies of a given field 
import numpy as np
import xarray as xr
import pandas as pd
import scipy.fft as fft 
# # OLR
# # remove 120-day running averages
# varn = 'olr'
# latsel = 90
# latflg = 'GfltG'

# # select the wanted time range and latitude range
# ds1 = xr.open_dataset("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+".6hr.1978to2022.nc")
# ds1 = ds1.sel(time=slice('1978-09-03', '2022-05-31'), lat=slice(latsel,-latsel))

# fltu = ds1[varn].rolling(time=120*4, center=False).mean()
# ds1 = ds1.sel(time=slice('1979-01-01', '2022-05-31'))
# ds1[varn] = ds1[varn] - fltu[120*4-1:-1,:,:].values
# ds1[varn] = - ds1[varn] / 3600   # convert the unit to W/m2
# print("removed 120-day running averages")
# print(varn)
# print("min: "+str(ds1[varn].min()))
# print("max: " + str(ds1[varn].max()))

# # remove the first 3 Fourier harmonics in 
# # which is the smoothed climatology daily average during year 1979-2014
# dstmp = ds1.sel(time=slice('1979-01-01','2014-12-31'))
# ds2 = dstmp.groupby("time.dayofyear").mean()
# del dstmp
# climvarn = ds2[varn].values
# climvarnfft = fft.rfftn(climvarn, axes=0)
# climvarnfft[3:,:,:] = 0.0
# smclimvarn = fft.irfftn(climvarnfft, axes=0)
# del climvarn
# del climvarnfft
# ds2[varn].values = smclimvarn
# del smclimvarn
# x = ds1[varn].groupby("time.dayofyear") - ds2[varn]
# del ds2
# ds1[varn].values = x.values

# ds1.to_netcdf("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+latflg+".6hr.1979to2022.nc")
# del ds1

# # U850
# # remove 120-day running averages
# varn = 'u850'
# latsel = 90
# latflg = 'GfltG'

# # select the wanted time range and latitude range
# ds1 = xr.open_dataset("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+".6hr.1978to2022.nc")
# ds1 = ds1.sel(time=slice('1978-09-03', '2022-05-31'), lat=slice(latsel,-latsel))

# fltu = ds1[varn].rolling(time=120*4, center=False).mean()
# ds1 = ds1.sel(time=slice('1979-01-01', '2022-05-31'))
# ds1[varn] = ds1[varn] - fltu[120*4-1:-1,:,:].values
# print("removed 120-day running averages")
# print(varn)
# print("min: "+str(ds1[varn].min()))
# print("max: " + str(ds1[varn].max()))

# # remove the first 3 Fourier harmonics in 
# # which is the smoothed climatology daily average during year 1979-2014
# dstmp = ds1.sel(time=slice('1979-01-01','2014-12-31'))
# ds2 = dstmp.groupby("time.dayofyear").mean()
# del dstmp
# climvarn = ds2[varn].values
# climvarnfft = fft.rfftn(climvarn, axes=0)
# climvarnfft[3:,:,:] = 0.0
# smclimvarn = fft.irfftn(climvarnfft, axes=0)
# del climvarn
# del climvarnfft
# ds2[varn].values = smclimvarn
# del smclimvarn
# x = ds1[varn].groupby("time.dayofyear") - ds2[varn]
# del ds2
# ds1[varn].values = x.values

# ds1.to_netcdf("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+latflg+".6hr.1979to2022.nc")

# del ds1

# U200
# remove 120-day running averages
varn = 'u200'
latsel = 90
latflg = 'GfltG'
dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'

# select the wanted time range and latitude range
flg = 'u200'

ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='by_coords')

ds1=ds1.rename({'u':flg,"longitude":"lon","latitude":"lat"})

ds1 = ds1.sel(time=slice('1978-09-03', '2022-05-31'), lat=slice(latsel,-latsel))

fltu = ds1[varn].rolling(time=120*4, center=False).mean()
ds1 = ds1.sel(time=slice('1979-01-01', '2022-05-31'))
ds1[varn] = ds1[varn] - fltu[120*4-1:-1,:,:].values
print("removed 120-day running averages")
print(varn)
print("min: "+str(ds1[varn].min()))
print("max: " + str(ds1[varn].max()))

# remove the first 3 Fourier harmonics in 
# which is the smoothed climatology daily average during year 1979-2014
dstmp = ds1.sel(time=slice('1979-01-01','2014-12-31'))
ds2 = dstmp.groupby("time.dayofyear").mean()
del dstmp
climvarn = ds2[varn].values
climvarnfft = fft.rfftn(climvarn, axes=0)
climvarnfft[3:,:,:] = 0.0
smclimvarn = fft.irfftn(climvarnfft, axes=0)
del climvarn
del climvarnfft
ds2[varn].values = smclimvarn
del smclimvarn
x = ds1[varn].groupby("time.dayofyear") - ds2[varn]
del ds2
ds1[varn].values = x.values

y = np.sum(np.isnan(x.values))

if y==0 : 
    ds1.to_netcdf("/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5."+varn+latflg+".6hr.1979to2022.nc")


