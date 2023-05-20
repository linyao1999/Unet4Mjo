import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import numpy as np
import sys
# import netCDF4 as nc
from saveNCfile import savenc
from saveNCfile_for_activations import savenc_for_activations
from Reloaddata_loader_loop import load_test_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
import os 
from datetime import date 

# Updates Nov 13, 2022: Add 2dMonte
# input: 6 global maps at three time steps
# output: RMMERA5 at one time step

# parameters to be set
testystat = 2015  # validation starts
nmaps = 19  # the number of variables we include in the input
nmapsnorm = 18  # the number of variables need to be normalized
lat_lim = 20  # maximum latitude in degree

testyend = 2020  # validation ends
num_epochs = 100   # how many loops we want to train the model. 
Nsamp = 100  # MC runs applied to forecast of testing data


datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
Fn1 = datadir+'ERA5.u200GfltG.day.1979to2020.nc'
Fn2 = datadir+'ERA5.u850GfltG.day.1979to2020.nc'
Fn3 = datadir+'ERA5.olrGfltG.day.1979to2020.nc'

Fn4 = datadir+'ERA5.tcwvGfltG.day.1979to2020.nc'
Fn5 = datadir+'ERA5.v200GfltG.day.1979to2020.nc'
Fn6 = datadir+'ERA5.T200GfltG.day.1979to2020.nc'
Fn7 = datadir+'ERA5.prepGfltG.day.1979to2020.nc'

Fn8 = datadir+'ERA5.u500GfltG.day.1979to2020.nc'
Fn9 = datadir+'ERA5.v500GfltG.day.1979to2020.nc'
Fn10 = datadir+'ERA5.v850GfltG.day.1979to2020.nc'
Fn11 = datadir+'ERA5.Z200GfltG.day.1979to2020.nc'
Fn12 = datadir+'ERA5.Z500GfltG.day.1979to2020.nc'
Fn13 = datadir+'ERA5.Z850GfltG.day.1979to2020.nc'
Fn14 = datadir+'ERA5.T500GfltG.day.1979to2020.nc'
Fn15 = datadir+'ERA5.T850GfltG.day.1979to2020.nc'
Fn16 = datadir+'ERA5.q200GfltG.day.1979to2020.nc'
Fn17 = datadir+'ERA5.q500GfltG.day.1979to2020.nc'
Fn18 = datadir+'ERA5.q850GfltG.day.1979to2020.nc'

# always be careful to let sst as the last input variable
Fn19 = datadir+'ERA5.sstGfltGmask0.day.1979to2020.nc'

# file name list
Fn = [Fn1,Fn2,Fn3,Fn4,Fn5,Fn6,Fn7,Fn8,Fn9,Fn10,Fn11,Fn12,Fn13,Fn14,Fn15,Fn16,Fn17,Fn18,Fn19]
# variable name list
vn = ['u200','u850','olr','tcwv','v200','T200','prep','u500','v500','v850','Z200','Z500','Z850','T500','T850','q200','q500','q850','sst']
Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1901to2020.nc'
### PATHS and FLAGS ###
# path_static_activations = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/activations_analysis'
# path_weights = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/weights_analysis/'
# # path_forecasts = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/forecasts_2deg/'
path_forecasts = './output/'
model_save = '/pscratch/sd/l/linyaoly/Unet4MJO/19variables_MCDO_36yr_filtered_trop/'
if not os.path.isdir(path_forecasts):
        os.makedirs(path_forecasts)
        print("create output directory")

if not os.path.isdir(model_save):
        os.makedirs(model_save)
        print("create model_save directory")

leadmjo = int(os.environ["lead30d"]) # lead for output (the MJO index)
nmem = int(os.environ["memlen"])  # the number of how many days we want to include into the input maps

print('leadmjo: '+str(leadmjo))
print('nmem: '+str(nmem))

mem_list = np.arange(nmem)

batch_size = 20
num_samples = 2
lambda_reg = 0.2

# model's hyperparameters
nhidden1=500
nhidden2=200
nhidden3=50

dimx = int(1 + 2 * int(lat_lim / 2))
dimy = 180

num_filters_enc = 64
num_filters_dec1 = 128
num_filters_dec2 = 192

featureDim=num_filters_dec2*dimx*dimy


class CNN(nn.Module):
    def __init__(self,imgChannels=nmaps*nmem, out_channels=2):
        super().__init__()
        self.input_layer = (nn.Conv2d(imgChannels, num_filters_enc, kernel_size=5, stride=1, padding='same'))
        self.hidden1 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
        self.hidden2 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))


        self.hidden5 = (nn.Conv2d(num_filters_dec1, num_filters_dec1, kernel_size=5, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(num_filters_dec2, num_filters_dec2, kernel_size=5, stride=1, padding='same' ))

        self.FC1 = nn.Linear(featureDim,nhidden1)
        self.FC2 = nn.Linear(nhidden1,nhidden2)
        self.FC3 = nn.Linear(nhidden2,nhidden3)
        self.FC4 = nn.Linear(nhidden3,out_channels)

        self.dropoutconv = nn.Dropout2d(p=0.1)
        self.dropoutline = nn.Dropout(p=0.2)

    def forward (self,x):

        x1 = F.relu (self.dropoutconv(self.input_layer(x)))
        x2 = F.relu (self.dropoutconv(self.hidden1(x1)))
        x3 = F.relu (self.dropoutconv(self.hidden2(x2)))
        x4 = F.relu (self.dropoutconv(self.hidden3(x3)))

        x5 = torch.cat ((F.relu(self.dropoutconv(self.hidden4(x4))),x3), dim =1)
        x6 = torch.cat ((F.relu(self.dropoutconv(self.hidden5(x5))),x2), dim =1)
        x6 = x6.view(-1,featureDim)
        x6 = F.relu(self.FC1(x6))
        x7 = F.relu(self.FC2(self.dropoutline(x6)))
        x8 = F.relu(self.FC3(self.dropoutline(x7)))

        out = (self.FC4(self.dropoutline(x8)))

        return out

net = CNN()

net.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('**** Number of Trainable Parameters in BNN')
count_parameters(net)

print('Model starts')

# torch.save(net.state_dict(), model_save+'predicted_MCDO_UNET_'+str(len(vn))+'mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput.pt')
net.load_state_dict(torch.load(model_save+'predicted_MCDO_UNET_'+str(len(vn))+'mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput.pt'))
net = net.eval()

RMMp = [[0,0]]
RMM_dis = [np.zeros([Nsamp,2])]
RMMt = [[0,0]]

for testyn in np.arange(testystat,testyend):
        print("test year is: " + str(testyn))

        psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(Fn,vn,Fnmjo,leadmjo,mem_list,testyn,lat_lim)
        psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

        for leveln in np.arange(0,nmapsnorm*nmem):
                M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
                STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
                psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

        psi_test_input_Tr_torch_norm[:,nmapsnorm*nmem:nmaps*nmem,None,:,:] = psi_test_input_Tr_torch[:,nmapsnorm*nmem:nmaps*nmem,None,:,:]

        psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

        print('shape of normalized input test',psi_test_input_Tr_torch.shape)
        print('shape of normalized label test',psi_test_label_Tr_torch.shape)
        ###############################################################################
        delta = date(testyn+1,1,1) - date(testyn,1,1)
        M = delta.days
        autoreg_pred = np.zeros([M,2]) # Nlat changed to 1 for hovmoller forecast
        autoreg_pred_dis = np.zeros([M,Nsamp,2]) # save each MCDO run
        autoreg_true = np.zeros([M,2])

        for k in range(0,M):
                # generate samples for each lead using slightly different 'models'
                inputx = psi_test_input_Tr_torch[k].reshape([1,nmaps*nmem,dimx,dimy]).cuda()
                net = net.eval()
                out = net(inputx).data.cpu().numpy()

                net = net.train()
                out_dis = np.array([net(inputx).data.cpu().numpy() for _ in range(Nsamp)]).squeeze()

                autoreg_pred[k,None,:] = out
                autoreg_pred_dis[k,None,:,:] = out_dis
                autoreg_true[k,None,:] = psi_test_label_Tr[k,:]

        RMMp = np.append(RMMp, autoreg_pred, axis=0)
        RMM_dis = np.append(RMM_dis, autoreg_pred_dis, axis=0)
        RMMt = np.append(RMMt, autoreg_true, axis=0)

tstat = np.datetime64(str(testystat)) + np.timedelta64(nmem-1,'D')
tend = np.datetime64(str(testyend)) + np.timedelta64(nmem-1,'D')

t = np.arange(tstat, tend, dtype='datetime64[D]')

# create prediction RMM time series
RMMp = xr.DataArray(
        data=RMMp[1:,:],
        dims=['time','mode'],
        coords=dict(
                time=t,
                mode=[0,1]
        ),
        attrs=dict(
                description='RMM prediction'
        ),
        name='RMMp',
)

# create prediction RMM distribution time series
RMMp_dis = xr.DataArray(
        data=RMM_dis[1:,:,:],
        dims=['time','N','mode'],
        coords=dict(
                time=t,
                N=np.arange(Nsamp),
                mode=[0,1]
        ),
        attrs=dict(
                description='RMM prediction distribution'
        ),
        name='RMMp_dis',
)

# create true RMM time series
RMMt = xr.DataArray(
        data=RMMt[1:,:],
        dims=['time','mode'],
        coords=dict(
                time=t,
                mode=[0,1]
        ),
        attrs=dict(
                description='RMM truth'
        ),
        name='RMMt',
)

ds = xr.merge([RMMp, RMMt, RMMp_dis])

ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+str(len(vn))+'mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem'+str(nmem)+'d.nc')



