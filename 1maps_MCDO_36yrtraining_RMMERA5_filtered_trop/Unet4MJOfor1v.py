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
from data_loader import load_test_data
from data_loader import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
import os 

# Updates Nov 25, 2022: use one variable to find minimum recipe
# variable order:
# 0: u200
# 1: u850
# 2: olr
# 3: tcwv
# 4: v200
# 5: T200
# 6: sst

# Updates Nov 13, 2022: Add 2dMonte
# input: 6 global maps at three time steps
# output: RMMERA5 at one time step


# parameters to be set
ysta = 1979  # training starts
testystat = 2015  # validation starts

# yend = 1980
# testyend = 2020
# num_epochs = 100
# trainN = 345
# M = 365

yend = 2015  # training ends
testyend = 2020  # validation ends
num_epochs = 100   # how many loops we want to train the model. 
trainN = 345  # 365 - batch_size
M = 365  

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
Fn1 = datadir+'ERA5.u200GfltG.day.1979to2022.nc'
Fn2 = datadir+'ERA5.u850GfltG.day.1979to2022.nc'
Fn3 = datadir+'ERA5.olrGfltG.day.1979to2022.nc'

Fn4 = datadir+'ERA5.tcwvGfltG.day.1979to2022.nc'
Fn5 = datadir+'ERA5.v200GfltG.day.1979to2022.nc'
Fn6 = datadir+'ERA5.T200GfltG.day.1979to2022.nc'
Fn7 = datadir+'ERA5.sstGfltGmask0.day.1979to2022.nc'

Fn8 = datadir+'ERA5.u500GfltG.day.1979to2022.nc'
Fn9 = datadir+'ERA5.v500GfltG.day.1979to2022.nc'
Fn10 = datadir+'ERA5.v850GfltG.day.1979to2022.nc'
Fn11 = datadir+'ERA5.Z200GfltG.day.1979to2022.nc'
Fn12 = datadir+'ERA5.Z500GfltG.day.1979to2022.nc'
Fn13 = datadir+'ERA5.Z850GfltG.day.1979to2022.nc'
Fn14 = datadir+'ERA5.T500GfltG.day.1979to2022.nc'
Fn15 = datadir+'ERA5.T850GfltG.day.1979to2022.nc'
Fn16 = datadir+'ERA5.q200GfltG.day.1979to2022.nc'
Fn17 = datadir+'ERA5.q500GfltG.day.1979to2022.nc'
Fn18 = datadir+'ERA5.q850GfltG.day.1979to2022.nc'
Fn19 = datadir+'ERA5.prepGfltG.day.1979to2022.nc'


# Fn4 = datadir+'ERA5.tcwv.day.1978to2022.nc'
# Fn5 = datadir+'ERA5.v200.day.1978to2022.nc'
# Fn6 = datadir+'ERA5.T200.day.1978to2022.nc'
# Fn7 = datadir+'ERA5.sstGfltGmask0.day.1979to2022.nc'
Fn = []
nmaps = 1  # the number of variables we include in the input
nmapsnorm = 1  # the number of variables need to be normalized

varn = os.environ["varname"]

if varn=="u200":
        Fn.append(Fn1)
elif varn == "u850":
        Fn.append(Fn2)
elif varn == "olr":
        Fn.append(Fn3)
elif varn == "tcwv":
        Fn.append(Fn4)
elif varn == "v200":
        Fn.append(Fn5)
elif varn == "T200":
        Fn.append(Fn6)
elif varn == "sst":
        Fn.append(Fn7)
        nmapsnorm = 0
elif varn == "u500":
        Fn.append(Fn8)
elif varn == "v500":
        Fn.append(Fn9)
elif varn == "v850":
        Fn.append(Fn10)
elif varn == "Z200":
        Fn.append(Fn11)
elif varn == "Z500":
        Fn.append(Fn12)
elif varn == "Z850":
        Fn.append(Fn13)
elif varn == "T500":
        Fn.append(Fn14)
elif varn == "T850":
        Fn.append(Fn15)
elif varn == "q200":
        Fn.append(Fn16)
elif varn == "q500":
        Fn.append(Fn17)
elif varn == "q850":
        Fn.append(Fn18)
elif varn == "prep":
        Fn.append(Fn19)

Fnmjo = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
### PATHS and FLAGS ###
# path_static_activations = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/activations_analysis'
# path_weights = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/weights_analysis/'
# # path_forecasts = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/U-NET/forecasts_2deg/'
path_forecasts = './output_dp0102_1dmem_' + varn + '/'
model_save = '/pscratch/sd/l/linyaoly/Unet4MJO/'
if not os.path.isdir(path_forecasts):
        os.makedirs(path_forecasts)
        print("create output directory")

leadmjo = int(os.environ["lead30d"]) # lead for output (the MJO index)
nmem = int(os.environ["memlen"])  # the number of how many days we want to include into the input maps

print('leadmjo: '+str(leadmjo))
print('nmem: '+str(nmem))
print('variable: '+ varn)

mem_list = np.arange(nmem)

batch_size = 20
num_samples = 2
lambda_reg = 0.2
knlat = 15 # added by lin 91->2deg # Nlat changed to 1 for hovmoller forecast
knlon = 180 # added by lin 180->2deg

# model's hyperparameters
nhidden1=500
nhidden2=200
nhidden3=50

dimx = 15
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
psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(Fn,Fnmjo,leadmjo,mem_list,testystat,varn)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmapsnorm*nmem):
  M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
  STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
  psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

psi_test_input_Tr_torch_norm[:,nmapsnorm*nmem:nmaps*nmem,None,:,:] = psi_test_input_Tr_torch[:,nmapsnorm*nmem:nmaps*nmem,None,:,:]
psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

# print('shape of normalized input test',psi_test_input_Tr_torch.shape)
# print('shape of normalized label test',psi_test_label_Tr_torch.shape)
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


def RK4step(net,input_batch):
 output_1,_,_,_,_,_,_ = net(input_batch.cuda())
 output_2,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_1)
 output_3,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_2)
 output_4,_,_,_,_,_,_ = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6
  


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

# # changed the following four lines by lin
# Act_encoder = np.zeros([num_epochs,nmaps*nmem,num_samples,64,knlat,knlon])   #### Last three: number of channels, Nalt, Nlon
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
     psi_train_input_Tr_torch, psi_train_label_Tr_torch = load_train_data(Fn,Fnmjo,leadmjo,mem_list,yn,varn)
     psi_train_input_Tr_torch_norm = np.zeros(np.shape(psi_train_input_Tr_torch))

     for levelnloop in np.arange(0,nmapsnorm*nmem):
        M_train_level = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        STD_train_level = torch.std(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        psi_train_input_Tr_torch_norm[:,levelnloop,None,:,:] = ((psi_train_input_Tr_torch[:,levelnloop,None,:,:]-M_train_level)/STD_train_level)

     psi_train_input_Tr_torch_norm[:,nmapsnorm*nmem:nmaps*nmem,None,:,:] = psi_train_input_Tr_torch[:,nmapsnorm*nmem:nmaps*nmem,None,:,:]
     psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr_torch_norm).float()

#      print('shape of normalized input test',psi_train_input_Tr_torch.shape)
#      print('shape of normalized label test',psi_train_label_Tr_torch.shape)

     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:]
        # print('shape of input', input_batch.shape)
        # print('shape of output', label_batch.shape)

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


     # add additional validation ##################
#    out = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,dim_lon,96]).cuda())

#    hidden_weights_encoder, hidden_weights_decoder1, final_weights_network = store_weights(net,epoch,hidden_weights_encoder, hidden_weights_decoder1, final_weights_network)

#    Act_encoder, Act_decoder1, Act_decoder2, output_training = store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training, epoch,out,x1,x2,x3,x4,x5,x6)

print('Finished Training')

net = net.eval()
torch.save(net.state_dict(), model_save+'predicted_MCDO_UNET_7mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput.pt')

print('BNN Model Saved')

if (FLAGS_ACTIVATIONS_DUMP ==1):
 savenc_for_activations(Act_encoder, Act_decoder1, Act_decoder2,output_training,2,num_epochs,4,num_samples,64,128,192,dim_lon,96,path_static_activations+'BNN_UNET_no_sponge_FFT_loss_Activations_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(leadmjo)+'.nc')

 print('Saved Activations for BNN')

if (FLAGS_WEIGHTS_DUMP ==1):

 matfiledata = {}
 matfiledata[u'hidden_weights_encoder'] = hidden_weights_encoder
 matfiledata[u'hidden_weights_decoder'] = hidden_weights_decoder1
 matfiledata[u'final_layer_weights'] = final_weights_network
 hdf5storage.write(matfiledata, '.', path_weights+'BNN_RK4UNET_no_sponge_FFT_loss_Weights_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.mat', matlab_compatible=True)

 print('Saved Weights for BNN')

############# Auto-regressive prediction #####################
# STD_test_level1 = STD_test_level1.detach().cpu().numpy()
# M_test_level1 = M_test_level1.detach().cpu().numpy()


# STD_test_level2 = STD_test_level2.detach().cpu().numpy()
# M_test_level2 = M_test_level2.detach().cpu().numpy()


# STD_test_level3 = STD_test_level3.detach().cpu().numpy()
# M_test_level3 = M_test_level3.detach().cpu().numpy()

# STD_testlabel_level1 = STD_testlabel_level1.detach().cpu().numpy()
# M_testlabel_level1 = M_testlabel_level1.detach().cpu().numpy()


# STD_testlabel_level2 = STD_testlabel_level2.detach().cpu().numpy()
# M_testlabel_level2 = M_testlabel_level2.detach().cpu().numpy()


for testyn in np.arange(testystat,testyend):
        print("test year is: " + str(testyn))

        del psi_test_input_Tr_torch
        del psi_test_label_Tr_torch
        del psi_test_label_Tr

        psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(Fn,Fnmjo,leadmjo,mem_list,testyn,varn)
        psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

        for leveln in np.arange(0,nmapsnorm*nmem):
                M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
                STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
                psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)
        
        psi_test_input_Tr_torch_norm[:,nmapsnorm*nmem:nmaps*nmem,None,:,:] = psi_test_input_Tr_torch[:,nmapsnorm*nmem:nmaps*nmem,None,:,:]

        psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

        # print('shape of normalized input test',psi_test_input_Tr_torch.shape)
        # print('shape of normalized label test',psi_test_label_Tr_torch.shape)
        ###############################################################################

        autoreg_pred = np.zeros([M,2]) # Nlat changed to 1 for hovmoller forecast
        Nsamp = 100
        autoreg_pred_av = np.zeros([M,2])  # (step, RMM index)
        autoreg_pred_std = np.zeros([M,2])
        autoreg_true = np.zeros([M,2])

        for k in range(0,M):
                # generate samples for each lead using slightly different 'models'
                inputx = psi_test_input_Tr_torch[k].reshape([1,nmaps*nmem,dimx,dimy]).cuda()
                net = net.train()
                out_dis = np.array([net(inputx).data.cpu().numpy() for _ in range(Nsamp)]).squeeze()
                net = net.eval()
                out = net(inputx).data.cpu().numpy()
                # out = (net(psi_test_input_Tr_torch[k].reshape([1,nmaps*nmem,dimx,dimy]).cuda())) # Nlat changed to 1 for hovmoller forecast
                net = net.train()

                print('out_dis shape: '+str(out_dis.shape))
                autoreg_pred_av[k,None,:] = out_dis.mean(axis=0)
                autoreg_pred_std[k,None,:] = out_dis.std(axis=0)
                # autoreg_pred[k,None,:] = (out.detach().cpu().numpy())
                autoreg_pred[k,None,:] = out
                autoreg_true[k,None,:] = psi_test_label_Tr[k,:]

        #    autoreg_pred[k,:,:,:] = autoreg_pred[k,:,:,:]*STD_test+M_test

        # out = (net(torch.from_numpy(((autoreg_pred[k-1,:,:,:])).reshape([1,3,91,180])).float().cuda())) # Nlat changed to 1 for hovmoller forecast
        # autoreg_pred[k,:] = (out.detach().cpu().numpy())
        # autoreg_pred[k,:] = autoreg_pred[k,:]

        # autoreg_pred_level1_denorm = autoreg_pred[:,0]*STD_testlabel_level1+M_testlabel_level1
        # autoreg_pred_level2_denorm = autoreg_pred[:,1]*STD_testlabel_level2+M_testlabel_level2
        # autoreg_pred_level1_denorm = np.reshape(autoreg_pred_level1_denorm,(np.size(autoreg_pred_level1_denorm,0),1)) 
        # autoreg_pred_level2_denorm = np.reshape(autoreg_pred_level2_denorm,(np.size(autoreg_pred_level2_denorm,0),1))
        # autoreg_pred = np.concatenate((autoreg_pred_level1_denorm,autoreg_pred_level2_denorm),axis=1)

        print('autoreg_pred' + str(autoreg_pred[0,0]))
        print('autoreg_pred' + str(autoreg_pred.shape))

        np.savetxt(path_forecasts+'predicted_MCDO_UNET_'+varn+'trop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem'+str(nmem)+'d'+str(testyn)+'.csv', autoreg_pred, delimiter=",")
        np.savetxt(path_forecasts+'truth_MCDO_UNET_'+varn+'trop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem'+str(nmem)+'d'+str(testyn)+'.csv', autoreg_true, delimiter=",")
        np.savetxt(path_forecasts+'predicted_disav_MCDO_UNET_'+varn+'trop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem'+str(nmem)+'d'+str(testyn)+'.csv', autoreg_pred_av, delimiter=",")
        np.savetxt(path_forecasts+'predicted_disstd_MCDO_UNET_'+varn+'trop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_mem'+str(nmem)+'d'+str(testyn)+'.csv', autoreg_pred_std, delimiter=",")

        # savenc(autoreg_pred, path_forecasts+'predicted_UNET_u200u850olr_RMMERA5_lead'+str(leadmjo)+'.nc')
        # savenc(autoreg_true, path_forecasts+'truth_UNET_u200u850olr_RMMERA5_lead'+str(leadmjo)+'.nc')

