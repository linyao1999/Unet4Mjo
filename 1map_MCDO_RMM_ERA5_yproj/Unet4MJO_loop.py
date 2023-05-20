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
from data_loader_loop import load_test_data
from data_loader_loop import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
import os 
from datetime import date 
from projection import projection


# April 18: project OLR onto meridional wave structures
# Updates Nov 13, 2022: Add 2dMonte
# input: 6 global maps at three time steps
# output: RMMERA5 at one time step

# parameters to be set
ysta = int(os.environ["ysta_train"])  # training starts
yend = int(os.environ["yend_train"])  # training ends

testystat = int(os.environ["ysta_test"])  # validation starts
testyend = int(os.environ["yend_test"])  # validation ends

leadmjo = int(os.environ["lead30d"]) # lead for output (the MJO index)
nmem = int(os.environ["memlen"])  # the number of how many days we want to include into the input maps
print('leadmjo: '+str(leadmjo))
print('nmem: '+str(nmem))

zmode = int(os.environ["vertical_mode"])  # selected verticla mode 
m = int(os.environ["m"])  # number of meridional modes

lat_lim = int(os.environ["lat_lim"])  # maximum latitude in degree

nmaps = 1
num_epochs = 100   # how many loops we want to train the model. 
Nsamp = 100  # MC runs applied to forecast of testing data

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
Fn = datadir+'ERA5.olrGfltG.day.1901to2020.nc'
vn = 'olr'

# Fnmjo = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1901to2020.nc'

path_forecasts = './output/'
model_save = '/pscratch/sd/l/linyaoly/Unet4MJO/1map_MCDO_RMM_yproj/'
if not os.path.isdir(path_forecasts):
        os.makedirs(path_forecasts)
        print("create output directory")

if not os.path.isdir(model_save):
        os.makedirs(model_save)
        print("create model_save directory")


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

psi_train_input_Tr_torch, psi_train_label_Tr_torch, trainN = load_train_data(Fnmjo,leadmjo,mem_list,ysta,yend,zmode,m,lat_lim)
psi_train_input_Tr_torch_norm = np.zeros(np.shape(psi_train_input_Tr_torch))

for levelnloop in np.arange(0,nmem):
        M_train_level = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        STD_train_level = torch.std(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        psi_train_input_Tr_torch_norm[:,levelnloop,None,:,:] = ((psi_train_input_Tr_torch[:,levelnloop,None,:,:]-M_train_level)/STD_train_level)

psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_train_input_Tr_torch.shape)
print('shape of normalized label test',psi_train_label_Tr_torch.shape)


psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M  = load_test_data(Fnmjo,leadmjo,mem_list,testystat,testyend,zmode,m,lat_lim)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmem):
        M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)
###############################################################################


for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    
        for step in range(0,trainN-batch_size,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:]
            print('shape of input', input_batch.shape)
            print('shape of output', label_batch.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= net(input_batch.cuda())

            loss = loss_fn(output, label_batch.cuda())

            loss.backward()
            optimizer.step()
            output_val= net(psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,nmaps*nmem,dimx,dimy]).cuda()) # Nlat changed to 1 for hovmoller forecast
            val_loss = loss_fn(output_val, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,2]).cuda()) # Nlat changed to 1 for hovmoller forecast
            # print statistics

            if step % 50 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, loss))
                print('[%d, %5d] val_loss: %.3f' %
                    (epoch + 1, step + 1, val_loss))
                running_loss = 0.0

            del input_batch
            del label_batch


print('Finished Training')

net = net.eval()
torch.save(net.state_dict(), model_save+'predicted_MCDO_UNET_1map'+str(lat_lim)+'deg_RMM_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+'_zmode'+str(zmode)+'_nmem'+str(nmem)+'.pt')

# net.load_state_dict(torch.load(model_save+'predicted_MCDO_UNET_1map_RMM_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+'_zmode'+str(zmode)+'_nmem'+str(nmem)+'.pt'))
# net.eval()

print('BNN Model Saved')


autoreg_pred = np.zeros([M,2]) # Nlat changed to 1 for hovmoller forecast
autoreg_pred_dis = np.zeros([M,Nsamp,2]) # save each MCDO run
autoreg_true = np.zeros([M,2])

for k in range(0,M):
        # generate samples for each lead using slightly different 'models'
        inputx = psi_test_input_Tr_torch[k].reshape([1,nmem,dimx,dimy]).cuda()

        net = net.eval()
        out = net(inputx).data.cpu().numpy()

        net = net.train()
        out_dis = np.array([net(inputx).data.cpu().numpy() for _ in range(Nsamp)]).squeeze() 

        autoreg_pred[k,None,:] = out
        autoreg_pred_dis[k,None,:,:] = out_dis
        autoreg_true[k,:] = psi_test_label_Tr[k,:]

print('out shape: ', np.shape(out))
print('out_dis shape: ', np.shape(out_dis))

t0 = np.datetime64(str(testystat)) + np.timedelta64(mem_list[-1], 'D')
tb = np.datetime64(str(testyend))

t1 = np.arange(t0,tb)

# create prediction RMM time series
RMMp = xr.DataArray(
        data=autoreg_pred,
        dims=['time','mode'],
        coords=dict(
                time=t1,
                mode=[0,1]
        ),
        attrs=dict(
                description='RMM prediction'
        ),
        name='RMMp',
)

# create prediction RMM distribution time series
RMMp_dis = xr.DataArray(
        data=autoreg_pred_dis,
        dims=['time','N','mode'],
        coords=dict(
                time=t1,
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
        data=autoreg_true,
        dims=['time','mode'],
        coords=dict(
                time=t1,
                mode=[0,1]
        ),
        attrs=dict(
                description='RMM truth'
        ),
        name='RMMt',
)

ds = xr.merge([RMMp, RMMt, RMMp_dis])

ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_olr'+str(lat_lim)+'deg_RMMERA5_'+str(m)+'modes_lead'+str(leadmjo)+'_dailyinput_'+str(ysta)+'to'+str(yend)+'_mem'+str(nmem)+'d.nc')

