# calculate RMM index
# use filtered anomalies from getfiltereddata.py
import numpy as np 
import pandas as pd 
import xarray as xr 

# read OLR anomalies
fnolr = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.olrGfltG.day.1979to2022.nc'
dsolr = xr.open_dataset(fnolr)
dsolr = dsolr.sel(lat=slice(15,-15))
# averaged over latitude
avolr = dsolr['olr'].mean(dim="lat")

# read u850 anomalies
fnu850 = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.u850GfltG.day.1979to2022.nc'
dsu850 = xr.open_dataset(fnu850)
dsu850 = dsu850.sel(lat=slice(15,-15))
# averaged over latitude
avu850 = dsu850['u850'].mean(dim="lat")

# read u200 anomalies
fnu200 = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.u200GfltG.day.1979to2022.nc'
dsu200 = xr.open_dataset(fnu200)
dsu200 = dsu200.sel(lat=slice(15,-15))
# averaged over latitude
avu200 = dsu200['u200'].mean(dim="lat")
# time, lon

# Normalize with the temporal variance during 1979-2014 at different longitude
# select 1979-2014 OLR, time x lon
tmp = avolr.sel(time=slice('1979-01-01','2014-12-31'))
stdolr = tmp.std(dim="time")
stdolr = stdolr.mean()
avolrnm = avolr / stdolr
del tmp 

# select 1979-2014 u850, time x lon
tmp = avu850.sel(time=slice('1979-01-01','2014-12-31'))
stdu850 = tmp.std(dim="time")
stdu850 = stdu850.mean()
avu850nm = avu850 / stdu850
del tmp 

# select 1979-2014 u200, time x lon
tmp = avu200.sel(time=slice('1979-01-01','2014-12-31'))
stdu200 = tmp.std(dim="time")
stdu200 = stdu200.mean()
avu200nm = avu200 / stdu200

RMM_field = xr.concat([avolrnm,avu850nm,avu200nm], dim="lon")

from eofs.xarray import Eof  
solver = Eof(RMM_field.sel(time=slice('1979-01-01','2014-12-31')), center=False)
EOF_RMM_field = solver.eofs(neofs=2)
EOF_RMM_field = EOF_RMM_field.transpose()

eigenvalue1 = solver.eigenvalues(neigs=2)

# Change the Sign of EOF to be consistent with WH04
ieof1max, ieof2max = EOF_RMM_field[0:180,:].argmax(dim="lon")
lonmaxeof1 = EOF_RMM_field.lon[ieof1max]
lonmaxeof2 = EOF_RMM_field.lon[ieof2max]

if (lonmaxeof1 >= 100) & (lonmaxeof1 <= 180) :
    EOF_RMM_field[:,0] = - EOF_RMM_field[:,0]

if (lonmaxeof2 >= 120) & (lonmaxeof2 <= 220) :
    EOF_RMM_field[:,1] = - EOF_RMM_field[:,1]

# project the whole dataset onto the EOF during 1979-2014
PC_RMM_field = RMM_field.dot(EOF_RMM_field)
tmp = PC_RMM_field / np.sqrt(eigenvalue1)

PC_RMM_field = (tmp - tmp.mean(dim='time')) / tmp.std(dim='time')

# ds = pd.DataFrame(PC_RMM_field, columns=['RMM1','RMM2'])
# ds['Date'] = PC_RMM_field['time']
# ds.to_csv('/global/homes/l/linyaoly/ERA5/reanalysis/RMMERA5only/ERA5_RMM_ERA5only.csv', index=False)

EOF_RMM_field.name = 'EOF'
EOF_RMM_field.to_netcdf('RMMeof_ERA5_daily.nc')

PC_RMM_field.name = 'RMM'
PC_RMM_field.to_netcdf('RMM_ERA5_daily.nc')