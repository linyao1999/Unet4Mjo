import numpy as np
import xarray as xr
import pandas as pd

# dirn = '/global/cscratch1/sd/linyaoly/ERA5/reanalysis/'

# flg = 'olr'
# varn = 'ttr'
# ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='by_coords')

# ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})
# ds1[flg].astype(float)
# # ds1['time'] = pd.to_datetime(ds1.time)

# ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.nc')

# del ds1

flg = 'u200'
varn = 'u'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='by_coords')

ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})
ds1[flg].astype(float)
# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.nc')

del ds1

# flg = 'u850'
# varn = 'u'
# ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='by_coords')

# ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})
# ds1[flg].astype(float)
# # ds1['time'] = pd.to_datetime(ds1.time)

# ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.nc')

# del ds1

dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'tcwv'
varn = 'tcwv'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({"longitude":"lon","latitude":"lat"})

ds1[flg].astype(float)

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.1978to2022.nc')
del ds1


dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'v200'
varn = 'v'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})

ds1[flg].astype(float)

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.1978to2022.nc')

del ds1

dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
flg = 'T200'
varn = 't'
ds1 = xr.open_mfdataset(dirn+flg+'_2deg*.nc',combine='nested',concat_dim="time")

ds1=ds1.rename({varn:flg,"longitude":"lon","latitude":"lat"})

ds1[flg].astype(float)

# ds1['time'] = pd.to_datetime(ds1.time)

ds1.to_netcdf(dirn+'ERA5.'+flg+'.6hr.1978to2022.nc')
del ds1

