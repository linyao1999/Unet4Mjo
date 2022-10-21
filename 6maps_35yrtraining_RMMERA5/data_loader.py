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

def load_test_data(Fn1,Fn2,Fn3,Fn4,Fn5,Fn6,Fnmjo,leadmjo,mem_list):
  # open six datasets
  FF1=xr.open_mfdataset(Fn1) #changed by lin
  FF2=xr.open_mfdataset(Fn2) #changed by lin
  FF3=xr.open_mfdataset(Fn3) #changed by lin
  FF4=xr.open_mfdataset(Fn4) #changed by lin
  FF5=xr.open_mfdataset(Fn5) #changed by lin
  FF6=xr.open_mfdataset(Fn6) #changed by lin

  # calculate daily average
  psi1=np.asarray(FF1['u'].groupby(FF1.time.dt.date).mean('time'))  # u200
  psi2=np.asarray(FF2['u'].groupby(FF2.time.dt.date).mean('time'))  # u850
  psi3=np.asarray(FF3['ttr'].groupby(FF3.time.dt.date).mean('time'))  # olr
  psi4=np.asarray(FF4['tp'].groupby(FF4.time.dt.date).mean('time'))  # prep
  psi5=np.asarray(FF5['v'].groupby(FF5.time.dt.date).mean('time'))  # v200
  psi6=np.asarray(FF6['t'].groupby(FF6.time.dt.date).mean('time'))  # T200

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
  
  psi_test_input = np.concatenate((psi11,psi21,psi31,psi41,psi51,psi61),axis=1)
  print('shape of psi_test_input: '+str(psi_test_input.shape))

  df1 = pd.read_csv(Fnmjo[0])
  df2 = pd.read_csv(Fnmjo[1])
  df = pd.concat([df1,df2],ignore_index=True)
  pc1 = np.asarray(df.PC1)
  pc2 = np.asarray(df.PC2)
 
  lat=np.asarray(FF1['latitude'])
  lon=np.asarray(FF1['longitude'])

  Nlat=np.size(lat,0)
  Nlon=np.size(lon,0)

  pc1 = np.reshape(pc1,(np.size(pc1,0),1)) 
  pc2 = np.reshape(pc2,(np.size(pc2,0),1)) 

  pc = np.concatenate((pc1,pc2),axis=1)
  
  print('pc shape')
  print(str(pc.shape))

  psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:]

  psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),6*nmem,Nlat,Nlon])   # 6 input maps
  psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

  psi_test_input_Tr = psi_test_input
  psi_test_label_Tr = psi_test_label

  # print('test input has nan: '+str(np.sum(np.isnan(psi_test_input))))
  # print('test label has nan: '+str(np.sum(np.isnan(psi_test_label))))

  # why loop like below?
  # for k in range(0,np.size(psi_test_input,0)):
  #   psi_test_input_Tr[k,0,:,:] = psi_test_input[k,0,:,:]
  #   psi_test_input_Tr[k,1,:,:] = psi_test_input[k,1,:,:]
  #   psi_test_input_Tr[k,2,:,:] = psi_test_input[k,2,:,:]
  #   psi_test_input_Tr[k,3,:,:] = psi_test_input[k,3,:,:]
  #   psi_test_input_Tr[k,4,:,:] = psi_test_input[k,4,:,:]
  #   psi_test_input_Tr[k,5,:,:] = psi_test_input[k,5,:,:]
  #   psi_test_input_Tr[k,6,:,:] = psi_test_input[k,6,:,:]
  #   psi_test_input_Tr[k,7,:,:] = psi_test_input[k,7,:,:]
  #   psi_test_input_Tr[k,8,:,:] = psi_test_input[k,8,:,:]
  #   psi_test_input_Tr[k,9,:,:] = psi_test_input[k,9,:,:]
  #   psi_test_input_Tr[k,10,:,:] = psi_test_input[k,10,:,:]
  #   psi_test_input_Tr[k,11,:,:] = psi_test_input[k,11,:,:]
  #   psi_test_input_Tr[k,12,:,:] = psi_test_input[k,12,:,:]
  #   psi_test_input_Tr[k,13,:,:] = psi_test_input[k,13,:,:]
  #   psi_test_input_Tr[k,14,:,:] = psi_test_input[k,14,:,:]
  #   psi_test_input_Tr[k,15,:,:] = psi_test_input[k,15,:,:]
  #   psi_test_input_Tr[k,16,:,:] = psi_test_input[k,16,:,:]
  #   psi_test_input_Tr[k,17,:,:] = psi_test_input[k,17,:,:]
    
  #   psi_test_label_Tr[k,0] = psi_test_label[k,0]
  #   psi_test_label_Tr[k,1] = psi_test_label[k,1]
    
  ## convert to torch tensor
  psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
  psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

  return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr


def load_train_data(loop1,loop2,loop3,loop4,loop5,loop6,loopmjo,leadmjo,mem_list):     ##### change the rest of the stuff in train
  
     # open six datasets
     File1=xr.open_mfdataset(loop1)
     File2=xr.open_mfdataset(loop2)
     File3=xr.open_mfdataset(loop3)
     File4=xr.open_mfdataset(loop4)
     File5=xr.open_mfdataset(loop5)
     File6=xr.open_mfdataset(loop6)

     # calculate daily average
     psi1=np.asarray(File1['u'].groupby(File1.time.dt.date).mean('time'))  # u200
     psi2=np.asarray(File2['u'].groupby(File2.time.dt.date).mean('time'))  # u850
     psi3=np.asarray(File3['ttr'].groupby(File3.time.dt.date).mean('time'))  # olr
     psi4=np.asarray(File4['tp'].groupby(File4.time.dt.date).mean('time'))  # prep
     psi5=np.asarray(File5['v'].groupby(File5.time.dt.date).mean('time'))  # v200
     psi6=np.asarray(File6['t'].groupby(File6.time.dt.date).mean('time'))  # T200

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
     
     psi_train_input = np.concatenate((psi11,psi21,psi31,psi41,psi51,psi61),axis=1)
     print('shape of psi_train_input: '+str(psi_train_input.shape))

     df1 = pd.read_csv(loopmjo[0])
     df2 = pd.read_csv(loopmjo[1])
     df = pd.concat([df1,df2],ignore_index=True)
     pc1 = np.asarray(df.PC1)
     pc2 = np.asarray(df.PC2)

     lat=np.asarray(File1['latitude'])
     lon=np.asarray(File1['longitude'])

     Nlat=np.size(lat,0)
     Nlon=np.size(lon,0)

     pc1 = np.reshape(pc1,(np.size(pc1,0),1))
     pc2 = np.reshape(pc2,(np.size(pc2,0),1))
     pc = np.concatenate((pc1,pc2),axis=1)

     psi_train_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays,:] 

     psi_train_input = np.reshape(psi_train_input,(ndays,6*nmem,Nlat,Nlon))
     psi_train_label = np.reshape(psi_train_label,(ndays,2))

     psi_train_label_Tr = psi_train_label
     psi_train_input_Tr = psi_train_input

     print('Train input', np.shape(psi_train_input_Tr))
     print('Train label', np.shape(psi_train_label_Tr)) 
    #  print('train input has nan: '+str(np.sum(np.isnan(psi_train_input))))
    #  print('train label has nan: '+str(np.sum(np.isnan(psi_train_label))))

     psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
     psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()  

     return psi_train_input_Tr_torch, psi_train_label_Tr_torch
