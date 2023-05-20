import sys
import numpy as np
import pandas as pd  
import xarray as xr 
import dask
from scipy import special
import math 

def projection(zmode=1, m=5, time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90]):
    # zmode: the vertical mode, default m = 1
    # m: wave truncation
    # time_range: the time range of the data used in training and validating the model
    # lat_range: the latitude range (y) of the data used in projection

    # read data
    fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.olrGfltG.day.1901to2020.nc'
    ds = xr.open_dataset(fn)

    ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1]))

    olr = ds1['olr'].values  # (time, lat, lon)

    return olr  # (time, m, lon)










