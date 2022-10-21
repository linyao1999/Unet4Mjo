import numpy as np
import xarray as xr
import pandas as pd

dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'tcwv'
varn = 'tcwv'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({"longitude":"lon","latitude":"lat"})

ds1=ds1.resample(time="1D").mean('time')

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.day.1978to2022.nc')
del ds1


dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'v200'
varn = 'v'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})

ds1=ds1.resample(time="1D").mean('time')

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.day.1978to2022.nc')

del ds1

dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'T200'
varn = 't'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})

ds1=ds1.resample(time="1D").mean('time')

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.day.1978to2022.nc')
del ds1

dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'sst'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({"longitude":"lon","latitude":"lat"})

ds1=ds1.resample(time="1D").mean('time')

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.day.1978to2022.nc')
del ds1

# remove climatology daily mean during 1979-2014
dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'sst'
varn = 'sst'
ds = xr.open_dataset(dirn+'ERA5.'+flg+'.day.1978to2022.nc')
dstmp = ds.sel(time=slice('1979-01-01','2014-12-31'))
ds2 = dstmp.groupby("time.dayofyear").mean()

x = ds[varn].groupby("time.dayofyear") - ds2[varn]
ds[varn].values = x.values 
ds = ds.sel(time=slice('1979-01-01','2022-05-31'))

dstmp = ds.sel(time=slice('1979-01-01','2014-12-31'))

# normalized sst 
msst = dstmp['sst'].mean(dim=None, skipna=True)
stdsst = dstmp['sst'].std(dim=None, skipna=True)
ds['sst'].values = (ds['sst'].values -  msst.values) / stdsst.values 
ds = ds.fillna(0)
ds.to_netcdf(dirn+'ERA5.'+flg+'GfltGmask0.day.1979to2022.nc')


