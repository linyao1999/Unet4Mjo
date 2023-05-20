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

def load_test_data(Fn,vn,Fnmjo,leadmjo,mem_list,yn,yend,lat_lim,mjo_ind):
    # set parameters
    nmem = len(mem_list)  # memory length
    ndays = int( ( np.datetime64(str(yend)+'-01-01') - np.datetime64(str(yn)+'-01-01') ) / np.timedelta64(1,'D') ) - mem_list[-1] # how many test samples 
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    psi = np.zeros((ndays,nmem*len(vn),dimx,dimy))

    print('mem_list: ', mem_list)
    print('len(vn):', len(vn))
    # print(str(len(vn)))

    # open each dataset and select 365 samples
    for ivn in np.arange(len(vn)):
        # read the #ivn variable in the variable list
        FF0 = xr.open_dataset(Fn[ivn])  
        # slice the data in the given year and given latitude range
        FF0 = FF0.sel(time=slice(str(yn)+'-01-01', str(yend-1)+'-12-31'), lat=slice(lat_lim,-lat_lim))

        # read the #ivn variable  (time,lat,lon)
        psi0 = np.asarray(FF0[vn[ivn]])

        del FF0

        # add memories from the first day of the year
        psi00 = np.zeros((ndays,nmem,np.size(psi0,1),np.size(psi0,2)))
        for i,memstp in zip(np.arange(nmem),mem_list):
            tmp = psi0[0+memstp:ndays+memstp,:,:]
            psi00[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

        psi[:,ivn*nmem:(ivn+1)*nmem,:,:] = psi00
        
        del psi0
        del psi00

    # combine all variables together
    psi_test_input = psi.copy()
    del psi

    print('combined input shape is: ' + str(np.shape(psi_test_input)))

    # read the MJO index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(yn)+'-01-01', str(yend)+'-03-01'))
    pc = np.asarray(FFmjo[mjo_ind])

    Nlat=dimx
    Nlon=dimy

    psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:]

    psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),len(vn)*nmem,Nlat,Nlon])   # vn input maps
    psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

    psi_test_input_Tr = psi_test_input
    psi_test_label_Tr = psi_test_label

    ## convert to torch tensor
    psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
    psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

    return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, ndays


def load_train_data(Fn,vn,Fnmjo,leadmjo,mem_list,yn,lat_lim,mjo_ind):   ##### change the rest of the stuff in train
    # load training data
    # set parameters
    nmem = len(mem_list)  # memory length
    ndays = int(pd.Timestamp(yn, 12, 31).dayofyear)  # how many samples in one 'year'  
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    psi = np.zeros((ndays,nmem*len(vn),dimx,dimy))
    
    # open each dataset and select 365 samples
    for ivn in np.arange(len(vn)):
        # read the #ivn variable in the variable list
        FF0 = xr.open_dataset(Fn[ivn])  
        # slice the data in the given year and given latitude range
        FF0 = FF0.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(lat_lim,-lat_lim))

        # read the #ivn variable
        psi0 = np.asarray(FF0[vn[ivn]])

        del FF0

        # add memories from the first day of the year
        psi00 = np.zeros((ndays,nmem,np.size(psi0,1),np.size(psi0,2)))
        for i,memstp in zip(np.arange(nmem),mem_list):
            tmp = psi0[0+memstp:ndays+memstp,:,:]
            psi00[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

        psi[:,ivn*nmem:(ivn+1)*nmem,:,:] = psi00

        del psi0
        del psi00

    psi_train_input = psi.copy()
    del psi

    print('combined train input shape is: ' + str(np.shape(psi_train_input)))

    # read the RMM index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'))

    pc = np.asarray(FFmjo[mjo_ind])

    Nlat=dimx
    Nlon=dimy

    psi_train_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:]

    psi_train_input_Tr=np.zeros([ndays,len(vn)*nmem,Nlat,Nlon])   # vn input maps
    psi_train_label_Tr=np.zeros([ndays,2])  # 2 PC labels

    psi_train_input_Tr = psi_train_input
    psi_train_label_Tr = psi_train_label

    ## convert to torch tensor
    psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
    psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()

    return psi_train_input_Tr_torch, psi_train_label_Tr_torch
