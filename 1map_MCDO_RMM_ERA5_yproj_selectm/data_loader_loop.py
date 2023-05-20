import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import numpy as np
import pandas as pd  
from saveNCfile import savenc
import xarray as xr 
import dask
from projection import projection

def load_test_data(Fnmjo,leadmjo,mem_list,testystat,testyend,zmode,m,mflg,lat_lim):
    # set parameters
    nmem = len(mem_list)  # memory length
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    # make projection  (time, lat, lon)
    olr_re = projection(zmode, m, mflg, [str(testystat)+'-01-01', str(testyend-1)+'-12-31'], [lat_lim,-lat_lim])

    ndays = int( ( np.datetime64(str(testyend)+'-01-01') - np.datetime64(str(testystat)+'-01-01') ) / np.timedelta64(1,'D') ) 

    psi_test_input = np.zeros((ndays-mem_list[-1],nmem,dimx,dimy))

    for i in range(ndays-mem_list[-1]):
        psi_test_input[i, :, :, :] = olr_re[i:i+nmem, :, :]


    print('combined input shape is: ' + str(np.shape(psi_test_input)))

    # read the RMM index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(testystat)+'-01-01', str(testyend)+'-03-31'))
    FFmjo.fillna(0)
    pc = np.asarray(FFmjo['RMM'])

    Nlat=dimx
    Nlon=dimy

    psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays-mem_list[-1],:]

    psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),nmem,Nlat,Nlon])   # vn input maps
    psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

    psi_test_input_Tr = psi_test_input
    psi_test_label_Tr = psi_test_label

    ## convert to torch tensor
    psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
    psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

    return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, ndays-mem_list[-1]


def load_train_data(Fnmjo,leadmjo,mem_list,ysta,yend,zmode,m,mflg,lat_lim):
    # set parameters
    nmem = len(mem_list)  # memory length
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    # make projection  (time, m, lon)
    olr_re = projection(zmode, m, mflg, [str(ysta)+'-01-01', str(yend-1)+'-12-31'], [lat_lim,-lat_lim])

    ndays = int( ( np.datetime64(str(yend)+'-01-01') - np.datetime64(str(ysta)+'-01-01') ) / np.timedelta64(1,'D') ) 

    print('ndays: ', ndays)
    print('mem_list[-1]', mem_list[-1])

    psi_train_input = np.zeros((ndays-mem_list[-1],nmem,dimx,dimy))

    for i in range(ndays-mem_list[-1]):
        psi_train_input[i, :, :, :] = olr_re[i:i+nmem, :, :]


    print('combined input shape is: ' + str(np.shape(psi_train_input)))

    # read the RMM index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(ysta)+'-01-01', str(yend)+'-03-31'))
    FFmjo.fillna(0)
    pc = np.asarray(FFmjo['RMM'])

    Nlat=dimx
    Nlon=dimy

    psi_train_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays-mem_list[-1],:]
    print('label shape is: ' + str(np.shape(psi_train_label)))

    psi_train_input_Tr=np.zeros([np.size(psi_train_input,0),nmem,Nlat,Nlon])   # vn input maps
    psi_train_label_Tr=np.zeros([np.size(psi_train_label,0),2])  # 2 PC labels

    psi_train_input_Tr = psi_train_input
    psi_train_label_Tr = psi_train_label

    ## convert to torch tensor
    psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
    psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()

    return psi_train_input_Tr_torch, psi_train_label_Tr_torch, ndays-mem_list[-1]
