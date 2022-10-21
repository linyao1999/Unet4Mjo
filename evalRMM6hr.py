# calculate BCC, RMSE, amplitude error and phase error only for MJO amplitude >1
import numpy as np 
import xarray as xr
import pandas as pd 
import os.path

############## BCC & RMSE & Amplitude error & Phase error ##############
def eval_error(ds0):  
    # Date	PC1	PC2	IniAmp	RMMp1	RMMp2	RMMt1	RMMt2
    pc1p = ds0.RMMp1.values  # predicted RMM1
    pc2p = ds0.RMMp2.values  # predicted RMM2
    pc1t = ds0.RMMt1.values  # truth RMM1
    pc2t = ds0.RMMt2.values  # truth RMM2

    # BCC
    a = sum(pc1p*pc1t+pc2p*pc2t)
    b = np.sqrt(sum(pc1t*pc1t+pc2t*pc2t))
    c = np.sqrt(sum(pc1p*pc1p+pc2p*pc2p))
    BCC = a/b/c 

    # RMSE
    d = (pc1t-pc1p)*(pc1t-pc1p)+(pc2t-pc2p)*(pc2t-pc2p)
    RMSE = np.sqrt(np.mean(d))

    # amplitude error 
    ampp = np.sqrt(pc1p*pc1p+pc2p*pc2p)
    ampt = np.sqrt(pc1t*pc1t+pc2t*pc2t)
    amp_err = np.mean(ampp-ampt)

    del a 
    del b
    del c 
    del d 
    
    # phase error 
    a = pc1t * pc2p - pc2t * pc1p 
    b = pc1t * pc1p + pc2t * pc2p
    c = a / b 
    d = np.arctan(c) * 180. / np.pi
    pha_err = np.mean(d)

    return BCC, RMSE, amp_err, pha_err

# set path and filenames 
dir = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/6maps_35yrtraining_RMMERA5_6hr/output/'  
flg1 = 'u200u850olrprepv200T200_'

###################################### get av ###################################################
###################################### get av ###################################################
###################################### get av ###################################################
# set year start and end
ystat = 2017
yend = 2020

########### calculate the initial MJO amplitude 2015-2019 ##########
RMMlist = []
for i in np.arange(ystat, yend):
    ds = pd.read_csv('/global/homes/l/linyaoly/ERA5/reanalysis/RMMERA56hr/'+str(i)+'.csv', header=0)
    RMMlist.append(ds.iloc[0:365*4,:])
    del ds

ds = pd.concat(RMMlist, axis=0, ignore_index=True)
ds = ds.iloc[:, 1:4]
pc1 = ds.PC1.values
pc2 = ds.PC2.values
initial_amp = np.sqrt(pc1 * pc1 + pc2 * pc2)   
ds['IniAmp'] = initial_amp
print("ds shape: " + str(ds.shape))
############# loop start to evaluate Unet for different lead and memory ###################
# set lead and memory
for nmem1 in ["1"]:
    BCC = np.zeros(17)
    RMSE = np.zeros(17)
    amp_err = np.zeros(17)
    pha_err = np.zeros(17)
    tmp = np.append(np.arange(1,17), [30])
    for k,flg in zip(np.arange(len(tmp)), tmp): # lead 
        ########### read PCs in prediction and truth ################
        fplist = []
        ftlist = []
        for j in np.arange(ystat, yend):
            dsp = pd.read_csv(dir+'predicted_UNET_'+flg1+'RMMERA56hr_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['RMMp1', 'RMMp2'])
            fplist.append(dsp)
            del dsp
            dst = pd.read_csv(dir+'truth_UNET_'+flg1+'RMMERA56hr_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['RMMt1', 'RMMt2'])
            ftlist.append(dst)
            del dst

        dsp = pd.concat(fplist, axis=0, ignore_index=True)

        dst = pd.concat(ftlist, axis=0, ignore_index=True)

        ############## select strong MJO #################
        ds0 = pd.concat([ds, dsp, dst], axis=1)  # Date	PC1	PC2	IniAmp	RMMp1	RMMp2	RMMt1	RMMt2
        ds0 = ds0.groupby('Date').mean()
        strongMJO = ds0[ds0['IniAmp']>=1.0]
        del ds0
        BCC[k], RMSE[k], amp_err[k], pha_err[k] = eval_error(strongMJO)
        del strongMJO
    eval = {'BCC':BCC, 'RMSE':RMSE, 'AmpErr':amp_err, 'PhaErr':pha_err}
    eval = pd.DataFrame(data=eval,index=tmp)
    eval.to_csv('./eval/UNET_'+flg1+'RMMERA56hr_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(ystat)+'to'+str(yend)+'av_averaged.csv', index_label='lead')
    del eval

print("dsp shape: " + str(dsp.shape))
###################################### get spread in years ###################################################
###################################### get spread in years ###################################################
###################################### get spread in years ###################################################
del dsp 
del dst 
del ds 

########### calculate the initial MJO amplitude 2015-2019 ##########
for i in np.arange(ystat, yend):
    ds = pd.read_csv('/global/homes/l/linyaoly/ERA5/reanalysis/RMMERA56hr/'+str(i)+'.csv', header=0)
    ds = ds.iloc[0:365*4, 1:4]

    pc1 = ds.PC1.values
    pc2 = ds.PC2.values
    initial_amp = np.sqrt(pc1 * pc1 + pc2 * pc2)   
    ds['IniAmp'] = initial_amp
    print("ds shape: " + str(ds.shape))
    ############# loop start to evaluate Unet for different lead and memory ###################
    # set lead and memory
    for nmem1 in ["1"]:
        BCC = np.zeros(17)
        RMSE = np.zeros(17)
        amp_err = np.zeros(17)
        pha_err = np.zeros(17)
        tmp = np.append(np.arange(1,17), [30])
        for j,flg in zip(np.arange(len(tmp)), tmp): # lead 
            ########### read PCs in prediction and truth ################
            dsp = pd.read_csv(dir+'predicted_UNET_'+flg1+'RMMERA56hr_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(i)+'.csv', header=None, names=['RMMp1', 'RMMp2'])
            dst = pd.read_csv(dir+'truth_UNET_'+flg1+'RMMERA56hr_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(i)+'.csv', header=None, names=['RMMt1', 'RMMt2'])
            ############## select strong MJO #################
            ds0 = pd.concat([ds, dsp, dst], axis=1)  # Date	PC1	PC2	IniAmp	RMMp1	RMMp2	RMMt1	RMMt2
            ds0 = ds0.groupby('Date').mean()
            strongMJO = ds0[ds0['IniAmp']>=1.0]
            print("strongMJO: "+ str(strongMJO.shape))
            del ds0
            BCC[j], RMSE[j], amp_err[j], pha_err[j] = eval_error(strongMJO)
            del strongMJO
        eval = {'BCC':BCC, 'RMSE':RMSE, 'AmpErr':amp_err, 'PhaErr':pha_err}
        eval = pd.DataFrame(data=eval,index=tmp)
        eval.to_csv('./eval/UNET_'+flg1+'RMMERA56hr_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(i)+'_averaged.csv', index_label='lead')
        del eval
    
    del ds

print("dsp shape: " + str(dsp.shape))





