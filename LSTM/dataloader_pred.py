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

# use prediction RMM as input 
# use observed RMM as output

def load_train_data(leadmjo,ysta=1979,yend=2014,nmem=300,nmemUnet=1,nvar=1,noutputmem=1,normflg=1):

    # the first date used in the training data
    datesta = np.datetime64(str(ysta)) + np.timedelta64(nmemUnet-1+leadmjo,'D')
    # the last date used in the training data without memories
    dateend = np.datetime64(str(yend)+'-12-31')
    # the last date used in the training data with memories
    dateend1 = dateend + np.timedelta64(nmem-1,'D')
    delta = dateend - datesta
    Ntrain = delta.astype(int) + 1 # time steps/number of samples used for training

    print('Ntrain in def: ', Ntrain)
    

    # ############ prepare training input data and labels #################
    train_input_data = np.empty((Ntrain, nmem, nvar*2))  
    train_model_label = np.empty((Ntrain, noutputmem, 2))  
 
    # ################ get the first input ###############################
    # get the matrix of ture RMM (day, nmem) at t
    frmmt = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
    dsrmmt = xr.open_dataset(frmmt)

    # get the matrix of predicted RMM at t+lead
    frmmp = '/global/homes/l/linyaoly/ERA5/reanalysis/'+'predicted_MCDO_UNET_19mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem1d.nc'
    dsrmmp = xr.open_dataset(frmmp)
    dsrmmp['RMM'] = dsrmmp['__xarray_dataarray_variable__']
    dsrmmp = dsrmmp.drop(['__xarray_dataarray_variable__'])

    # slice the training data at t-lead 
    dstmp02 = dsrmmp.loc[dict(time=slice((datesta - np.timedelta64(leadmjo,'D')), (dateend1 - np.timedelta64(leadmjo,'D'))))]

    RMMp1 = dstmp02['RMM'][:,0].values  # (day)
    RMMp2 = dstmp02['RMM'][:,1].values  # (day)

    for i in range(Ntrain):
        train_input_data[i,:,0] = RMMp1[i:i+nmem]
        train_input_data[i,:,1] = RMMp2[i:i+nmem]

    # slice the training data
    dstmp01 = dsrmmt.loc[dict(time=slice((datesta + np.timedelta64(nmem-1,'D')), dateend1))]
    RMMt1 = dstmp01['RMM'][:,0].values  # (day)
    RMMt2 = dstmp01['RMM'][:,1].values  # (day)

    # add memories to the model label
    for i in range(Ntrain):
        train_model_label[i,:,0] = RMMt1[i]
        train_model_label[i,:,1] = RMMt2[i]

    if normflg:
        # normalize input_data and labels
        train_input_data_norm = np.empty((Ntrain, nmem, nvar*2))  # (Nsamples, channels, nmem)
        train_model_label_norm = np.empty((Ntrain, noutputmem, 2))  # (Nsampels, channels, nmem)

        for i in range(nvar*2):
            M_train = np.mean(train_input_data[:,:,i])
            std_train = np.std(train_input_data[:,:,i])

            train_input_data_norm[:,:,i,None] = (train_input_data[:,:,i,None] - M_train) / std_train

        del M_train
        del std_train

        for i in range(2):
            M_train = np.mean(train_model_label[:,:,i])
            std_train = np.std(train_model_label[:,:,i])

            train_model_label_norm[:,:,i,None] = (train_model_label[:,:,i,None] - M_train) / std_train
            # train_model_label_norm[:,:,i,None] = train_model_label[:,:,i,None] 

        ## convert to torch tensor
        train_input_data_norm_torch = torch.from_numpy(train_input_data_norm).float()
        train_model_label_norm_torch = torch.from_numpy(train_model_label_norm).float()
    else:
        ## convert to torch tensor
        train_input_data_norm_torch = torch.from_numpy(train_input_data).float()
        train_model_label_norm_torch = torch.from_numpy(train_model_label).float()

    return Ntrain, train_input_data_norm_torch, train_model_label_norm_torch



def load_test_data(leadmjo,ysta=2015,yend=2018,nmem=300,nmemUnet=1,nvar=1,noutputmem=1,normflg=1):

    # the first date used in the testing data
    datesta = np.datetime64(str(ysta)) + np.timedelta64(nmemUnet-1+leadmjo,'D')
    # the last date used in the testing data without memories
    dateend = np.datetime64(str(yend)+'-12-31')
    # the last date used in the testing data with memories
    dateend1 = dateend + np.timedelta64(nmem-1,'D')
    delta = dateend - datesta
    Ntest = delta.astype(int) + 1 # time steps/number of samples used for testing

    print('Ntest in def: ', Ntest)
    

    # ############ prepare testing input data and labels #################
    test_input_data = np.empty((Ntest, nmem, nvar*2))  
    test_model_label = np.empty((Ntest, noutputmem, 2))  
 
    # ################ get the first input ###############################
    # get the matrix of ture RMM (day, nmem) at t
    frmmt = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
    dsrmmt = xr.open_dataset(frmmt)

    # get the matrix of predicted RMM at t+lead
    frmmp = '/global/homes/l/linyaoly/ERA5/reanalysis/'+'predicted_MCDO_UNET_19mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem1d.nc'
    dsrmmp = xr.open_dataset(frmmp)
    dsrmmp['RMM'] = dsrmmp['__xarray_dataarray_variable__']
    dsrmmp = dsrmmp.drop(['__xarray_dataarray_variable__'])

    # slice the testing data at t-lead 
    dstmp02 = dsrmmp.loc[dict(time=slice((datesta - np.timedelta64(leadmjo,'D')), (dateend1 - np.timedelta64(leadmjo,'D'))))]

    RMMp1 = dstmp02['RMM'][:,0].values  # (day)
    RMMp2 = dstmp02['RMM'][:,1].values  # (day)

    for i in range(Ntest):
        test_input_data[i,:,0] = RMMp1[i:i+nmem]
        test_input_data[i,:,1] = RMMp2[i:i+nmem]

    # slice the testing data
    dstmp01 = dsrmmt.loc[dict(time=slice((datesta + np.timedelta64(nmem-1,'D')), dateend1))]
    RMMt1 = dstmp01['RMM'][:,0].values  # (day)
    RMMt2 = dstmp01['RMM'][:,1].values  # (day)

    # add memories to the model label
    for i in range(Ntest):
        test_model_label[i,:,0] = RMMt1[i]
        test_model_label[i,:,1] = RMMt2[i]

    if normflg:
        # normalize input_data and labels
        test_input_data_norm = np.empty((Ntest, nmem, nvar*2))  # (Nsamples, channels, nmem)
        test_model_label_norm = np.empty((Ntest, noutputmem, 2))  # (Nsampels, channels, nmem)

        for i in range(nvar*2):
            M_test = np.mean(test_input_data[:,:,i])
            std_test = np.std(test_input_data[:,:,i])

            test_input_data_norm[:,:,i,None] = (test_input_data[:,:,i,None] - M_test) / std_test

        del M_test
        del std_test
        
        M_test1 = np.mean(test_model_label[:,:,0])
        std_test1 = np.std(test_model_label[:,:,0])
        test_model_label_norm[:,:,0,None] = (test_model_label[:,:,0,None] - M_test1) / std_test1

        M_test2 = np.mean(test_model_label[:,:,1])
        std_test2 = np.std(test_model_label[:,:,1])
        test_model_label_norm[:,:,1,None] = (test_model_label[:,:,1,None] - M_test2) / std_test2

        ## convert to torch tensor
        test_input_data_norm_torch = torch.from_numpy(test_input_data_norm).float()
        test_model_label_norm_torch = torch.from_numpy(test_model_label_norm).float()
    else:
        M_test1 = 0
        M_test2 = 0
        std_test1 = 1
        std_test2 = 1 
        ## convert to torch tensor
        test_input_data_norm_torch = torch.from_numpy(test_input_data).float()
        test_model_label_norm_torch = torch.from_numpy(test_model_label).float()

    return Ntest, M_test1, M_test2, std_test1, std_test2, test_input_data_norm_torch, test_model_label_norm_torch
