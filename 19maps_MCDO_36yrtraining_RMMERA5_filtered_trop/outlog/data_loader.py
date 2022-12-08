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

def load_test_data(Fn,Fnmjo,leadmjo,mem_list,yn):
  # open 7 daily datasets and select one year data
  FF1 = xr.open_dataset(Fn[0]) # u200
  FF1 = FF1.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF2 = xr.open_dataset(Fn[1]) # u850
  FF2 = FF2.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF3 = xr.open_dataset(Fn[2]) # olr
  FF3 = FF3.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF4 = xr.open_dataset(Fn[3]) # tcwv
  FF4 = FF4.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF5 = xr.open_dataset(Fn[4]) # v200
  FF5 = FF5.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF6 = xr.open_dataset(Fn[5]) # T200
  FF6 = FF6.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF7 = xr.open_dataset(Fn[6]) # sst
  FF7 = FF7.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  
  FF8 = xr.open_dataset(Fn[7]) # sst
  FF8 = FF8.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF9 = xr.open_dataset(Fn[8]) # sst
  FF9 = FF9.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF10 = xr.open_dataset(Fn[9]) # sst
  FF10 = FF10.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF11 = xr.open_dataset(Fn[10]) # sst
  FF11 = FF11.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF12 = xr.open_dataset(Fn[11]) # sst
  FF12 = FF12.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF13 = xr.open_dataset(Fn[12]) # sst
  FF13 = FF13.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF14 = xr.open_dataset(Fn[13]) # sst
  FF14 = FF14.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF15 = xr.open_dataset(Fn[14]) # sst
  FF15 = FF15.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF16 = xr.open_dataset(Fn[15]) # sst
  FF16 = FF16.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF17 = xr.open_dataset(Fn[16]) # sst
  FF17 = FF17.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF18 = xr.open_dataset(Fn[17]) # sst
  FF18 = FF18.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF19 = xr.open_dataset(Fn[18]) # sst
  FF19 = FF19.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))

  FFmjo = xr.open_dataset(Fnmjo)


  FFmjo = FFmjo.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'))
  FF1.fillna(0)
  FF2.fillna(0)
  FF3.fillna(0)
  FF4.fillna(0)
  FF5.fillna(0)
  FF6.fillna(0)
  FF7.fillna(0)
  FF8.fillna(0)
  FF9.fillna(0)
  FF10.fillna(0)
  FF11.fillna(0)
  FF12.fillna(0)
  FF13.fillna(0)
  FF14.fillna(0)
  FF15.fillna(0)
  FF16.fillna(0)
  FF17.fillna(0)
  FF18.fillna(0)
  FF19.fillna(0)

  FFmjo.fillna(0)
    
  psi1 = np.asarray(FF1['u200'])  # u200
  psi2 = np.asarray(FF2['u850'])  # u850
  psi3 = np.asarray(FF3['olr'])  # olr
  psi4 = np.asarray(FF4['tcwv'])  # tcwv
  psi5 = np.asarray(FF5['v200'])  # v200
  psi6 = np.asarray(FF6['T200'])  # T200
  psi7 = np.asarray(FF7['prep'])  # sst
  psi8 = np.asarray(FF8['u500'])  # sst
  psi9 = np.asarray(FF9['v500'])  # sst
  psi10 = np.asarray(FF10['v850'])  # sst
  psi11 = np.asarray(FF11['Z200'])  # sst
  psi12 = np.asarray(FF12['Z500'])  # sst
  psi13 = np.asarray(FF13['Z850'])  # sst
  psi14 = np.asarray(FF14['T500'])  # sst
  psi15 = np.asarray(FF15['T850'])  # sst
  psi16 = np.asarray(FF16['q200'])  # sst
  psi17 = np.asarray(FF17['q500'])  # sst
  psi18 = np.asarray(FF18['q850'])  # sst
  psi19 = np.asarray(FF19['sst'])  # sst

  pc = np.asarray(FFmjo['RMM'])

  # add memories
  nmem = len(mem_list)
  ndays = 365   # how many samples in one 'year'

  psi11 = np.zeros((ndays,nmem,np.size(psi1,1),np.size(psi1,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi1[0+memstp:ndays+memstp,:,:]
    psi11[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 
  print('shape of psi11: '+str(psi11.shape))

  psi21 = np.zeros((ndays,nmem,np.size(psi2,1),np.size(psi2,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi2[0+memstp:ndays+memstp,:,:]
    psi21[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi31 = np.zeros((ndays,nmem,np.size(psi3,1),np.size(psi3,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi3[0+memstp:ndays+memstp,:,:]
    psi31[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi41 = np.zeros((ndays,nmem,np.size(psi4,1),np.size(psi4,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi4[0+memstp:ndays+memstp,:,:]
    psi41[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi51 = np.zeros((ndays,nmem,np.size(psi5,1),np.size(psi5,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi5[0+memstp:ndays+memstp,:,:]
    psi51[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi61 = np.zeros((ndays,nmem,np.size(psi6,1),np.size(psi6,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi6[0+memstp:ndays+memstp,:,:]
    psi61[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 
  
  psi71 = np.zeros((ndays,nmem,np.size(psi7,1),np.size(psi7,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi7[0+memstp:ndays+memstp,:,:]
    psi71[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi_test_input = np.concatenate((psi11,psi21,psi31,psi41,psi51,psi61,psi71),axis=1)
  print('shape of psi_test_input: '+str(psi_test_input.shape))
 
  lat=np.asarray(FF1['lat'])
  lon=np.asarray(FF1['lon'])

  Nlat=np.size(lat,0)
  Nlon=np.size(lon,0)

  psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:]

  psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),7*nmem,Nlat,Nlon])   # 7 input maps
  psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

  psi_test_input_Tr = psi_test_input
  psi_test_label_Tr = psi_test_label

  ## convert to torch tensor
  psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
  psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

  return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr


def load_train_data(Fn,Fnmjo,leadmjo,mem_list,yn):     ##### change the rest of the stuff in train
  # open 7 daily datasets and select one year data
  FF1 = xr.open_dataset(Fn[0]) # u200
  FF1 = FF1.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF2 = xr.open_dataset(Fn[1]) # u850
  FF2 = FF2.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF3 = xr.open_dataset(Fn[2]) # olr
  FF3 = FF3.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF4 = xr.open_dataset(Fn[3]) # tcwv
  FF4 = FF4.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF5 = xr.open_dataset(Fn[4]) # v200
  FF5 = FF5.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF6 = xr.open_dataset(Fn[5]) # T200
  FF6 = FF6.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FF7 = xr.open_dataset(Fn[6]) # sst
  FF7 = FF7.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'), lat=slice(15,-15))
  FFmjo = xr.open_dataset(Fnmjo)
  FFmjo = FFmjo.sel(time=slice(str(yn)+'-01-01', str(yn+1)+'-03-01'))
  FF1.fillna(0)
  FF2.fillna(0)
  FF3.fillna(0)
  FF4.fillna(0)
  FF5.fillna(0)
  FF6.fillna(0)
  FF7.fillna(0)
  FFmjo.fillna(0)
  
  psi1=np.asarray(FF1['u200'])  # u200
  psi2=np.asarray(FF2['u850'])  # u850
  psi3=np.asarray(FF3['olr'])  # olr
  psi4=np.asarray(FF4['tcwv'])  # tcwv
  psi5=np.asarray(FF5['v200'])  # v200
  psi6=np.asarray(FF6['T200'])  # T200
  psi7=np.asarray(FF7['sst'])  # sst
  pc  =np.asarray(FFmjo['RMM'])

  # add memories
  nmem = len(mem_list)
  ndays = 365

  psi11 = np.zeros((ndays,nmem,np.size(psi1,1),np.size(psi1,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi1[0+memstp:ndays+memstp,:,:]
    psi11[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 
  print('shape of psi11: '+str(psi11.shape))

  psi21 = np.zeros((ndays,nmem,np.size(psi2,1),np.size(psi2,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi2[0+memstp:ndays+memstp,:,:]
    psi21[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi31 = np.zeros((ndays,nmem,np.size(psi3,1),np.size(psi3,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi3[0+memstp:ndays+memstp,:,:]
    psi31[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi41 = np.zeros((ndays,nmem,np.size(psi4,1),np.size(psi4,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi4[0+memstp:ndays+memstp,:,:]
    psi41[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi51 = np.zeros((ndays,nmem,np.size(psi5,1),np.size(psi5,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi5[0+memstp:ndays+memstp,:,:]
    psi51[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi61 = np.zeros((ndays,nmem,np.size(psi6,1),np.size(psi6,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi6[0+memstp:ndays+memstp,:,:]
    psi61[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 
  
  psi71 = np.zeros((ndays,nmem,np.size(psi7,1),np.size(psi7,2)))
  for i,memstp in zip(np.arange(nmem),mem_list):
    tmp = psi7[0+memstp:ndays+memstp,:,:]
    psi71[:,i,None,:,:] = np.reshape(tmp,(np.size(tmp,0),1,np.size(tmp,1),np.size(tmp,2))) 

  psi_train_input = np.concatenate((psi11,psi21,psi31,psi41,psi51,psi61,psi71),axis=1)
  print('shape of psi_train_input: '+str(psi_train_input.shape))
 
  lat=np.asarray(FF1['lat'])
  lon=np.asarray(FF1['lon'])

  Nlat=np.size(lat,0)
  Nlon=np.size(lon,0)

  psi_train_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:]

  psi_train_input_Tr = np.zeros([ndays,7*nmem,Nlat,Nlon])   # 7 input maps
  psi_train_label_Tr = np.zeros([ndays,2])  # 2 PC labels

  psi_train_input_Tr = psi_train_input
  psi_train_label_Tr = psi_train_label

  ## convert to torch tensor
  psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
  psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()
  print('Train input', np.shape(psi_train_input_Tr))
  print('Train label', np.shape(psi_train_label_Tr)) 

  return psi_train_input_Tr_torch, psi_train_label_Tr_torch