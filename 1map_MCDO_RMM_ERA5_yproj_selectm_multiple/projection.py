import sys
import numpy as np
import pandas as pd  
import xarray as xr 
import dask
from scipy import special
import math 

def projection(zmode=1, m=2, mflg='1pls', time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90]):
    # zmode: the vertical mode, default m = 1
    # m: wave truncation
    # time_range: the time range of the data used in training and validating the model
    # lat_range: the latitude range (y) of the data used in projection

    # parameters
    N = 1e-2  # buoyancy frequency
    H = 1.6e4  # tropopause height
    beta= 2.28e-11  # variation of coriolis parameter with latitude
    g = 9.8  # gravity acceleration 
    theta0 = 300  # surface potential temperature
    c = N * H / np.pi / zmode # gravity wave speed
    L = np.sqrt(c / beta)  # horizontal scale 

    # read data; any file with olr[time, lat, lon]
    fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.olrGfltG.day.1901to2020.nc'
    ds = xr.open_dataset(fn)

    ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1]))

    olr = ds1['olr'].values  # (time, lat, lon)
    lat = ds1['lat']
    lon = ds1['lon'].values
    time = ds1['time'].values

    # define y = lat * 110 km / L
    y = lat.values * 110 * 1000 / L # dimensionless

    # define meridianol wave structures
    phi = []

    # define which equatorial wave is included in the reconstructed map
    # m is analogous to a meridional wavenumber
    # m = 0: Kelvin wave
    # m = 2: Rossby wave
    
    if mflg=='odd':
        m_list = np.arange(1,m,2)  
    elif mflg=='even':
        m_list = np.arange(0,m,2)
    elif mflg=='all':
        m_list = np.arange(m)
    elif mflg=='one':
        m_list = [m-1]
    elif mflg=='1pls':
        m_list = [0,m-1]
    else:
        print('wrong m flag!')
        exit()

    for i in m_list:
        p = special.hermite(i)
        Hm = p(y)
        phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

        phi.append(np.reshape(phim, (1, len(y), 1)))

    # projection coefficients
    olrm = []

    dy = (lat[0].values - lat[1].values) * 110 * 1000 / L 

    for i in range(len(m_list)):
        um = np.sum(olr * phi[i] * dy, axis=1, keepdims=True)  # (time, 1, lon)
        olrm.append(um)

    # reconstruction 
    olr_re = np.zeros(np.shape(olr))  # (time, lat, lon)

    for i in range(len(m_list)):
        olr_re = olr_re + olrm[i] * phi[i]

    return olr_re  # (time, lat, lon)










