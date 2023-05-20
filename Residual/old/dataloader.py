import os

# import pytorch libraries to build nn
import torch 

# import basic libraries
import numpy as np 
import sys 
import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
from datetime import date 

def load_train_data(leadmjo,ysta=1979,yend=2014,nmem=300,nmemUnet=1,nvar=2,noutputmem=1,normflg=0):

    # the first date used in the training data
    datesta = np.datetime64(str(ysta)) + np.timedelta64(nmemUnet-1,'D')
    # the last date used in the training data without memories
    dateend = np.datetime64(str(yend)+'-12-31')
    # the last date used in the training data with memories
    dateend1 = dateend + np.timedelta64(nmem-1,'D')

    delta = dateend - datesta
    Ntrain = delta.astype(int)  # time steps/number of samples used for training

    print('Ntrain in def: ', Ntrain)
    
    # ############ prepare training input data and labels #################
    train_input_data = np.empty((Ntrain, nvar*2, nmem))  # (Nsamples, channels, nmem)
    train_model_label = np.empty((Ntrain, 2, noutputmem))  # (Nsampels, channels, nmem)

    # get the matrix of ture RMM (day, nmem) at t
    frmmt = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
    dsrmmt = xr.open_dataset(frmmt)

    # slice the training data
    dstmp = dsrmmt.loc[dict(time=slice(datesta, dateend1))]

    RMMt1 = dstmp['RMM'][:,0]  # (day)
    RMMt2 = dstmp['RMM'][:,1]  # (day)

    print('total training truth samples: ')
    print(len(RMMt1))
    print('The first truth is at time step:')
    print(RMMt1.time[0])
    print('The last truth is at time step:')
    print(RMMt1.time[-1])

    # get the matrix of predicted RMM at t+lead
    frmmp = '/global/homes/l/linyaoly/ERA5/reanalysis/'+'predicted_MCDO_UNET_19mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem1d.nc'
    dsrmmp = xr.open_dataset(frmmp)
    dsrmmp['RMM'] = dsrmmp['__xarray_dataarray_variable__']
    dsrmmp = dsrmmp.drop(['__xarray_dataarray_variable__'])

    # slice the training data
    dstmp1 = dsrmmp.loc[dict(time=slice(datesta, dateend1))]

    RMMp1 = dstmp1['RMM'][:,0]  # (day)
    RMMp2 = dstmp1['RMM'][:,1]  # (day)

    print('total training prediction samples: ')
    print(len(RMMp1))
    print('The first forecast is done at time step:')
    print(RMMp1.time[0])
    print('The last forecast is done at time step:')
    print(RMMp1.time[-1])

    if (len(RMMp1) != len(RMMt1)):
        print('The lengths of input variables are not equal!')
        quit()

    # add memories to the input data

    for i in range(Ntrain):
        train_input_data[i,0,:] = RMMt1[i:i+nmem].values
        train_input_data[i,1,:] = RMMt2[i:i+nmem].values
        train_input_data[i,2,:] = RMMp1[i:i+nmem].values
        train_input_data[i,3,:] = RMMp2[i:i+nmem].values

    del dstmp
    del RMMt1
    del RMMt2

    # get the matrix of the labels 
    # slice the training data at t+lead to calculate residual of truth - prediction
    dstmp2 = dsrmmt.loc[dict(time=slice((datesta + np.timedelta64(leadmjo,'D')), (dateend1 + np.timedelta64(leadmjo,'D'))))]
    RMM1 = dstmp2['RMM'][:,0]  # (day)
    RMM2 = dstmp2['RMM'][:,1]  # (day)

    resi1 = RMM1.values - RMMp1.values
    resi2 = RMM2.values - RMMp2.values

    # add memories to the model label

    for i in range(Ntrain):
        train_model_label[i,0,:] = resi1[i+nmem-noutputmem:i+nmem]
        train_model_label[i,1,:] = resi2[i+nmem-noutputmem:i+nmem]

    if normflg:
        # normalize input_data and labels
        train_input_data_norm = np.empty((Ntrain, nvar*2, nmem))  # (Nsamples, channels, nmem)
        train_model_label_norm = np.empty((Ntrain, 2, noutputmem))  # (Nsampels, channels, nmem)

        for i in range(nvar*2):
            M_train = np.mean(train_input_data[:,i,:])
            std_train = np.std(train_input_data[:,i,:])

            train_input_data_norm[:,i,None,:] = (train_input_data[:,i,None,:] - M_train) / std_train

        for i in range(2):
            # M_train = np.mean(train_model_label[:,i,:])
            # std_train = np.std(train_model_label[:,i,:])

            # train_model_label_norm[:,i,None,:] = (train_model_label[:,i,None,:] - M_train) / std_train
            train_model_label_norm[:,i,None,:] = train_model_label[:,i,None,:] 

        ## convert to torch tensor
        train_input_data_norm_torch = torch.from_numpy(train_input_data_norm).float()
        train_model_label_norm_torch = torch.from_numpy(train_model_label_norm).float()
    else:
        ## convert to torch tensor
        train_input_data_norm_torch = torch.from_numpy(train_input_data).float()
        train_model_label_norm_torch = torch.from_numpy(train_model_label).float()

    return Ntrain, train_input_data_norm_torch, train_model_label_norm_torch



def load_test_data(leadmjo,ysta=2015,yend=2018,nmem=300,nmemUnet=1,nvar=2,noutputmem=1,normflg=0):

    # the first date used in the testing data
    datesta = np.datetime64(str(ysta)) + np.timedelta64(nmemUnet-1,'D')
    # the last date used in the testing data without memories
    dateend = np.datetime64(str(yend)+'-12-31')
    # the last date used in the testing data with memories
    dateend1 = dateend + np.timedelta64(nmem-1,'D')

    delta = dateend - datesta
    Ntest = delta.astype(int)  # time steps/number of samples used for testing

    # ############ prepare testing input data and labels #################
    test_input_data = np.empty((Ntest, nvar*2, nmem))  # (Nsamples, channels, nmem)
    test_model_label = np.empty((Ntest, 2, noutputmem))  # (Nsampels, channels, nmem)

    # get the matrix of ture RMM (day, nmem) at t
    frmmt = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
    dsrmmt = xr.open_dataset(frmmt)

    # slice the testing data
    dstmp = dsrmmt.loc[dict(time=slice(datesta, dateend1))]

    RMMt1 = dstmp['RMM'][:,0]  # (day)
    RMMt2 = dstmp['RMM'][:,1]  # (day)

    print('total testing truth samples: ')
    print(len(RMMt1))
    print('The first truth is at time step:')
    print(RMMt1.time[0])
    print('The last truth is at time step:')
    print(RMMt1.time[-1])

    # get the matrix of predicted RMM at t+lead
    frmmp = '/global/homes/l/linyaoly/ERA5/reanalysis/'+'predicted_MCDO_UNET_19mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem1d.nc'
    dsrmmp = xr.open_dataset(frmmp)
    dsrmmp['RMM'] = dsrmmp['__xarray_dataarray_variable__']
    dsrmmp = dsrmmp.drop(['__xarray_dataarray_variable__'])

    # slice the testing data
    dstmp1 = dsrmmp.loc[dict(time=slice(datesta, dateend1))]

    RMMp1 = dstmp1['RMM'][:,0]  # (day)
    RMMp2 = dstmp1['RMM'][:,1]  # (day)

    print('total testing prediction samples: ')
    print(len(RMMp1))
    print('The first forecast is done at time step:')
    print(RMMp1.time[0])
    print('The last forecast is done at time step:')
    print(RMMp1.time[-1])

    if (len(RMMp1) != len(RMMt1)):
        print('The lengths of input variables are not equal!')
        quit()

    # add memories to the input data

    for i in range(Ntest):
        test_input_data[i,0,:] = RMMt1[i:i+nmem].values
        test_input_data[i,1,:] = RMMt2[i:i+nmem].values
        test_input_data[i,2,:] = RMMp1[i:i+nmem].values
        test_input_data[i,3,:] = RMMp2[i:i+nmem].values

    del dstmp
    del RMMt1
    del RMMt2

    # get the matrix of the labels 
    # slice the testing data at t+lead to calculate residual of truth - prediction
    dstmp2 = dsrmmt.loc[dict(time=slice((datesta + np.timedelta64(leadmjo,'D')), (dateend1 + np.timedelta64(leadmjo,'D'))))]
    RMM1 = dstmp2['RMM'][:,0]  # (day)
    RMM2 = dstmp2['RMM'][:,1]  # (day)

    resi1 = RMM1.values - RMMp1.values
    resi2 = RMM2.values - RMMp2.values

    # add memories to the model label

    for i in range(Ntest):
        test_model_label[i,0,:] = resi1[i+nmem-noutputmem:i+nmem]
        test_model_label[i,1,:] = resi2[i+nmem-noutputmem:i+nmem]

    if normflg:
        # normalize input_data 
        test_input_data_norm = np.empty((Ntest, nvar*2, nmem))  # (Nsamples, channels, nmem)
        test_model_label_norm = np.zeros((Ntest, 2, noutputmem))  # (Nsampels, channels, nmem)

        for i in range(nvar*2):
            M_test = np.mean(test_input_data[:,i,:])
            std_test = np.std(test_input_data[:,i,:])

            test_input_data_norm[:,i,None,:] = (test_input_data[:,i,None,:] - M_test) / std_test

        for i in range(2):
            # M_test = np.mean(test_model_label[:,i,:])
            # std_test = np.std(test_model_label[:,i,:])

            # test_model_label_norm[:,i,None,:] = (test_model_label[:,i,None,:] - M_test) / std_test
            test_model_label_norm[:,i,None,:] = test_model_label[:,i,None,:]

        ## convert to torch tensor
        test_input_data_norm_torch = torch.from_numpy(test_input_data_norm).float()
        test_model_label_norm_torch = torch.from_numpy(test_model_label_norm).float()
    else:
        ## convert to torch tensor
        test_input_data_norm_torch = torch.from_numpy(test_input_data).float()
        test_model_label_norm_torch = torch.from_numpy(test_model_label).float()

    return Ntest, test_input_data_norm_torch, test_model_label_norm_torch