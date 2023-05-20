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

# updates: May 11, 2023: upgrade codes to read os info
# Updates Nov 13, 2022: Add 2dMonte
# input: 6 global maps at three time steps
# output: RMMERA5 at one time step

# parameters to be set
ysta = int(os.environ["ysta_train"])  # training starts
yend = int(os.environ["yend_train"])  # training ends

testystat = int(os.environ["ysta_test"])  # validation starts
testyend = int(os.environ["yend_test"])  # validation ends

leadmjo = int(os.environ["leadmjo"]) # lead for output (the MJO index)
nmem = int(os.environ["nmem"])  # the number of how many days we want to include into the input maps
dmem = int(os.environ["dmem"]) # the interval of memory. dmem=1 means use every step in np.arange(nmem)
lat_lim = int(os.environ["lat_lim"])  # maximum latitude in degree

mjo_ind = os.environ["mjo_ind"]  # RMM or ROMI

print('MJO index', mjo_ind)
print('leadmjo: '+str(leadmjo))
print('nmem: '+str(nmem))
print('lat_lim: ', lat_lim)

nmaps = 1  # the number of variables we include in the input
nmapsnorm = 1  # the number of variables need to be normalized

num_epochs = 100   # how many loops we want to train the model. 
Nsamp = 100  # MC runs applied to forecast of testing data

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
# variable name list
vn = os.environ["varname"]

# file name list
Fn = []
fntemp = datadir + 'ERA5.' + vn + 'GfltG.day.1901to2020.nc'
Fn.append(fntemp)
del fntemp

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1901to2020.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2021.nc'
    
### PATHS and FLAGS ###
# path_static_activations = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/activations_analysis'
# path_weights = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/weights_analysis/'
# # path_forecasts = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/forecasts_2deg/'
path_forecasts = './output/'
model_save = '/pscratch/sd/l/linyaoly/Unet4MJO/1variable_MCDO_filtered_trop_May2023/'
if not os.path.isdir(path_forecasts):
        os.makedirs(path_forecasts)
        print("create output directory")

if not os.path.isdir(model_save):
        os.makedirs(model_save)
        print("create model_save directory")


mem_list = np.arange(0,nmem,dmem)

mem_list_len = len(mem_list)

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

### PATHS and FLAGS ###
# path_static_activations = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/activations_analysis'
# path_weights = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/weights_analysis/'
# # path_forecasts = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/forecasts_2deg/'
# path_forecasts = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/7maps_36yrtraining_RMMERA5_filtered_trop/output/'

FLAGS_WEIGHTS_DUMP=0
FLAGS_ACTIVATIONS_DUMP=0

##### prepare test data ###################################################
psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M  = load_test_data(Fn,[vn],Fnmjo,leadmjo,mem_list,testystat,testyend,lat_lim,mjo_ind)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmapsnorm*mem_list_len):
  M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
  STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
  psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

# psi_test_input_Tr_torch_norm[:,nmapsnorm*mem_list_len:nmaps*mem_list_len,None,:,:] = psi_test_input_Tr_torch[:,nmapsnorm*mem_list_len:nmaps*mem_list_len,None,:,:]
# psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)
###############################################################################

def store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training,epoch,out,x1,x2,x3,x4,x5,x6):

   Act_encoder[epoch,0,:,:,:,:] = x1.detach().cpu().numpy()
   Act_encoder[epoch,1,:,:,:,:] = x2.detach().cpu().numpy()
   Act_encoder[epoch,2,:,:,:,:] = x3.detach().cpu().numpy()
   Act_encoder[epoch,3,:,:,:,:] = x4.detach().cpu().numpy()

   Act_decoder1[epoch,:,:,:,:] = x5.detach().cpu().numpy()
   Act_decoder2[epoch,:,:,:,:] = x6.detach().cpu().numpy()


   output_training [epoch,:,:,:,:] = out.detach().cpu().numpy()

   return Act_encoder, Act_decoder1, Act_decoder2, output_training

def store_weights (net,epoch,hidden_weights_encoder,hidden_weights_decoder1,final_weights_network):

  hidden_weights_encoder[epoch,0,:,:,:,:] = net.hidden1.weight.data.cpu()
  hidden_weights_encoder[epoch,1,:,:,:,:] = net.hidden2.weight.data.cpu()
  hidden_weights_encoder[epoch,2,:,:,:,:] = net.hidden3.weight.data.cpu()
  hidden_weights_encoder[epoch,3,:,:,:,:] = net.hidden4.weight.data.cpu()


  hidden_weights_decoder1[epoch,:,:,:,:] = net.hidden5.weight.data.cpu()
  final_weights_network[epoch,:,:,:,:] = net.hidden6.weight.data.cpu()

  return hidden_weights_encoder, hidden_weights_decoder1, final_weights_network


def my_loss(output, target):

 loss1 = torch.mean((output-target)**2)
 loss = loss1
#  out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
#  target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

#  loss2 = torch.mean(torch.abs(out_fft[:,0,50:]-target_fft[:,0,50:]))

#  loss = (1-lamda_reg)*loss1 + lamda_reg*loss2 
 return loss



class CNN(nn.Module):
    def __init__(self,imgChannels=nmaps*mem_list_len, out_channels=2):
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

# # changed the following four lines by lin
# Act_encoder = np.zeros([num_epochs,nmaps*mem_list_len,num_samples,64,knlat,knlon])   #### Last three: number of channels, Nalt, Nlon
# Act_decoder1 = np.zeros([num_epochs,num_samples,128,knlat,knlon])
# Act_decoder2 = np.zeros([num_epochs,num_samples,192,knlat,knlon])
# # output_training = np.zeros([num_epochs,num_samples,nmaps, knlat, knlon])

# hidden_weights_encoder = np.zeros([num_epochs,4,64,64,5,5])
# hidden_weights_decoder1 = np.zeros([num_epochs,128,128,5,5])
# final_weights_network = np.zeros([num_epochs,3,192,5,5])


for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for yn in np.arange(ysta,yend):
        print('Training loop year '+str(yn))
        psi_train_input_Tr_torch, psi_train_label_Tr_torch = load_train_data(Fn,[vn],Fnmjo,leadmjo,mem_list,yn,lat_lim,mjo_ind)
        psi_train_input_Tr_torch_norm = np.zeros(np.shape(psi_train_input_Tr_torch))

        for levelnloop in np.arange(0,nmapsnorm*mem_list_len):
            M_train_level = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
            STD_train_level = torch.std(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
            psi_train_input_Tr_torch_norm[:,levelnloop,None,:,:] = ((psi_train_input_Tr_torch[:,levelnloop,None,:,:]-M_train_level)/STD_train_level)

        # psi_train_input_Tr_torch_norm[:,nmapsnorm*mem_list_len:nmaps*mem_list_len,None,:,:] = psi_train_input_Tr_torch[:,nmapsnorm*mem_list_len:nmaps*mem_list_len,None,:,:]
        # psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr_torch_norm).float()

        print('shape of normalized input test',psi_train_input_Tr_torch.shape)
        print('shape of normalized label test',psi_train_label_Tr_torch.shape)

        trainN = int(pd.Timestamp(yn, 12, 31).dayofyear)-batch_size

        for step in range(0,trainN,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:]
            print('shape of input', input_batch.shape)
            print('shape of output', label_batch.shape)

            # zero the parameter gradients
            optimizer.zero_grad()
            # print('check line 326')
            # forward + backward + optimize
    #        output,_,_,_,_,_,_ = net(input_batch.cuda())
            output= net(input_batch.cuda())

            # print('output has nan:'+str(sum(np.argwhere(np.isnan(np.asarray(output))))))
            # print('label has nan:'+str(sum(np.argwhere(np.isnan(np.asarray(label_batch.cuda()))))))
            loss = loss_fn(output, label_batch.cuda())
            # print('output has nan: '+str(np.sum(sum(torch.isnan(output)))))
            # print('label has nan: '+str(np.sum(sum(torch.isnan(label_batch.cuda())))))

            loss.backward()
            optimizer.step()
            output_val= net(psi_test_input_Tr_torch[step:num_samples+step].reshape([num_samples,nmaps*mem_list_len,dimx,dimy]).cuda()) # Nlat changed to 1 for hovmoller forecast
            val_loss = loss_fn(output_val, psi_test_label_Tr_torch[step:num_samples+step].reshape([num_samples,2]).cuda()) # Nlat changed to 1 for hovmoller forecast
            # print statistics

            if step % 50 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, loss))
                print('[%d, %5d] val_loss: %.3f' %
                    (epoch + 1, step + 1, val_loss))
                running_loss = 0.0

            del input_batch
            del label_batch


        # add additional validation ##################
    #    out = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,dim_lon,96]).cuda())

    #    hidden_weights_encoder, hidden_weights_decoder1, final_weights_network = store_weights(net,epoch,hidden_weights_encoder, hidden_weights_decoder1, final_weights_network)

    #    Act_encoder, Act_decoder1, Act_decoder2, output_training = store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training, epoch,out,x1,x2,x3,x4,x5,x6)

print('Finished Training')

net = net.eval()
torch.save(net.state_dict(), model_save+'predicted_MCDO_UNET_1map'+str(lat_lim)+'_'+mjo_ind+'ERA5_'+str(ysta)+'to'+str(yend)+'_lead'+str(leadmjo)+'_dailyinput_nmem'+str(nmem)+'_d'+str(dmem)+'.pt')

print('BNN Model Saved')

autoreg_pred = np.zeros([M,2]) # Nlat changed to 1 for hovmoller forecast
autoreg_pred_dis = np.zeros([M,Nsamp,2]) # save each MCDO run
autoreg_true = np.zeros([M,2])

for k in range(0,M):
        # generate samples for each lead using slightly different 'models'
        inputx = psi_test_input_Tr_torch[k].reshape([1,nmaps*mem_list_len,dimx,dimy]).cuda()

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
        name=mjo_ind+'p',
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
        name=mjo_ind+'p_dis',
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
        name=mjo_ind+'t',
)

ds = xr.merge([RMMp, RMMt, RMMp_dis])

ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_1map'+str(lat_lim)+'deg_'+mjo_ind+'ERA5_lead'+str(leadmjo)+'_dailyinput_'+str(ysta)+'to'+str(yend)+'_nmem'+str(nmem)+'_d'+str(dmem)+'.nc')

