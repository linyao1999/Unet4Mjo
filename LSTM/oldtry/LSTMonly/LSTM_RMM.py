import os

# import pytorch libraries to build nn
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

# import basic libraries
import numpy as np 
import sys 
import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
from datetime import date 
from dataloader import load_test_data
from dataloader import load_train_data

# ############  parameters ################
path_forecasts = './output/'
model_save = '/pscratch/sd/l/linyaoly/Unet4MJO/residual/19variables_MCDO_36yr_filtered_trop/'
if not os.path.isdir(path_forecasts):
        os.makedirs(path_forecasts)
        print("create output directory")

if not os.path.isdir(model_save):
        os.makedirs(model_save)
        print("create model_save directory")

leadmjo = int(os.environ["lead30d"]) # lead for prediction
ysta = 1979  # the first year we use to train the model 
yend = 2014  # the last year we use to train the model 
testystat = 2015  # the first year we use to test the model 
testyend = 2018  # the last year we use to test the model 

nmem = 300 
nmemUnet = 1  # how many memories we use in the Unet model to forecast RMM; this impacts the first time step we use in the NN
noutputmem = 1

normflg = 0  # whether normalize the input data
num_epochs = 100  # loop time
batch_size = 64
nvar = 3  # how many variables used as input
## get the normalized training inputs and labels 
Ntrain, train_input_data_norm_torch, train_model_label_norm_torch = load_train_data(leadmjo,ysta,yend,nmem,nmemUnet,nvar,noutputmem,normflg)

print('Ntrain: ', Ntrain)


# ################# define LSTM ##############
class LSTMnet(nn.Module):
    def __init__(self, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2*nvar, hidden_size=2)

    def forward(self, input):
        input_trans = input.view(-1, nmem, 2*nvar)  # (batch_size, nmem, modes)
        # lstm_out (batch_size, nmem, modes)
        lstm_out, (hn,cn) = self.lstm(input_trans)

        prediction = lstm_out[:, -1, None, :]

        return prediction


net = LSTMnet()
net.cuda()
# define the loss function to get the gradient
loss_fn = nn.MSELoss()

# define an optimizer to update the parameters
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Model starts')

loss_values = []

# ################## start training #####################################
for epoch in range(num_epochs):
    for step in range(0, Ntrain-batch_size, batch_size):
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))

        input_batch, label_batch = train_input_data_norm_torch[indices,:,:], train_model_label_norm_torch[indices,:,:]

        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(input_batch.cuda())

        loss = loss_fn(output, label_batch.reshape([batch_size,noutputmem,2]).cuda())

        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

        if step % 50 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, step + 1, loss))

        del input_batch
        del label_batch

print('Finished Training')
print('loss is:', loss_values)
net = net.eval()
torch.save(net.state_dict(), model_save+'residual_LSTM_MCDO_UNET_'+str(nvar)+'vari_19mapstrop_RMMERA5_36yr_lead'+str(leadmjo)+'_dailyinput_'+str(nmem)+'dmem.pt')

print('BNN Model Saved')


# ################# validation #############################
Ntest, M_test1, M_test2, std_test1, std_test2, test_input_data_norm_torch, test_model_label_norm_torch = load_test_data(leadmjo,testystat,testyend,nmem,nmemUnet,nvar,noutputmem,normflg)

print('Ntest:', Ntest)

resi_pred = np.zeros([Ntest,noutputmem,2])

for k in range(Ntest):
    inputx = test_input_data_norm_torch[k].reshape([1, nmem, nvar*2]).cuda()
    net = net.eval()
    out = net(inputx).data.cpu().numpy()

    resi_pred[k,None,:,:] = np.reshape(out,(1,noutputmem,2))

resi_pred1 = np.zeros([Ntest,noutputmem,2])

resi_pred1[:,:,0] = resi_pred[:,:,0] * std_test1 + M_test1
resi_pred1[:,:,1] = resi_pred[:,:,1] * std_test2 + M_test2

tstat = np.datetime64(str(testystat)) + np.timedelta64(nmem-1+nmemUnet-1+2*leadmjo,'D')
tend = np.datetime64(str(testyend)+'-12-31') + np.timedelta64(nmem+leadmjo,'D')

t = np.arange(tstat, tend, dtype='datetime64[D]')

print('M_test1: ', M_test1)
print('M_test2: ', M_test2)
print('std_test1: ', std_test1)
print('std_test2: ', std_test2)


# create prediction RMM time series
resi = xr.DataArray(
        data=np.squeeze(resi_pred1),
        dims=['time','mode'],
        coords=dict(
                time=t,
                mode=[0,1]
        ),
        attrs=dict(
                description='residual of RMM prediction at time'
        ),
)

ds = resi.to_dataset(name='resi')
ds.to_netcdf(path_forecasts+'residual_LSTM_MCDO_UNET_'+str(nvar)+'vari_RMMERA5_36yr_lead'+str(leadmjo)+'_norm'+str(normflg)+'_dailyinput_nmem'+str(nmem)+'nout'+str(noutputmem)+'d.nc')


