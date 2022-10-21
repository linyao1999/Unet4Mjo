# calculate BCC, RMSE, amplitude error and phase error only for MJO amplitude >1
import numpy as np 
import xarray as xr
import pandas as pd 
import os.path

############## BCC & RMSE & Amplitude error & Phase error ##############
def eval_error(ds0):
    pc1p = ds0.OMIp1.values  # predicted OMI1
    pc2p = ds0.OMIp2.values  # predicted OMI2
    pc1t = ds0.OMIt1.values  # truth OMI1
    pc2t = ds0.OMIt2.values  # truth OMI2

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
dir = '/global/homes/l/linyaoly/ERA5/script/Stability-Explanability/6maps_35yrtraining_OMI/output/'  
flg1 = 'u200u850olrprepv200T200_'

###################################### get av ###################################################
###################################### get av ###################################################
###################################### get av ###################################################
# set year start and end
ystat = 2016
yend = 2020

########### calculate the initial MJO amplitude 2015-2019 ##########
omilist = []
for i in np.arange(ystat, yend):
    ds = pd.read_csv('/global/homes/l/linyaoly/ERA5/reanalysis/omi/'+str(i)+'.csv', header=0)
    omilist.append(ds.iloc[0:365,:])
    del ds

ds = pd.concat(omilist, axis=0, ignore_index=True)
ds = ds.iloc[:, 1:4]

pc1 = ds.PC1.values
pc2 = ds.PC2.values
initial_amp = np.sqrt(pc1 * pc1 + pc2 * pc2)   
ds['IniAmp'] = initial_amp

############# loop start to evaluate Unet for different lead and memory ###################
# set lead and memory
for nmem1 in ["1","5","15","30"]:
    BCC = np.zeros(30)
    RMSE = np.zeros(30)
    amp_err = np.zeros(30)
    pha_err = np.zeros(30)
    for flg in np.arange(1,31): # lead 
        ########### read PCs in prediction and truth ################
        fplist = []
        ftlist = []
        for j in np.arange(ystat, yend):
            dsp = pd.read_csv(dir+'predicted_UNET_'+flg1+'OMI_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['OMIp1', 'OMIp2'])
            fplist.append(dsp)
            del dsp
            dst = pd.read_csv(dir+'truth_UNET_'+flg1+'OMI_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['OMIt1', 'OMIt2'])
            ftlist.append(dst)
            del dst

        dsp = pd.concat(fplist, axis=0, ignore_index=True)
        dst = pd.concat(ftlist, axis=0, ignore_index=True)

        ############## select strong MJO #################
        ds0 = pd.concat([ds, dsp, dst], axis=1)  # Date	PC1	PC2	IniAmp	OMIp1	OMIp2	OMIt1	OMIt2
        strongMJO = ds0[ds0['IniAmp']>=1.0]
        del ds0
        BCC[flg-1], RMSE[flg-1], amp_err[flg-1], pha_err[flg-1] = eval_error(strongMJO)
        del strongMJO
    eval = {'BCC':BCC, 'RMSE':RMSE, 'AmpErr':amp_err, 'PhaErr':pha_err}
    eval = pd.DataFrame(data=eval,index=np.arange(1,31))
    eval.to_csv('./eval/UNET_'+flg1+'OMI_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(ystat)+'to'+str(yend)+'av.csv', index_label='lead')
    del eval

###################################### get spread in years ###################################################
###################################### get spread in years ###################################################
###################################### get spread in years ###################################################

########### calculate the initial MJO amplitude 2015-2019 ##########
for i in np.arange(ystat, yend):
    ds = pd.read_csv('/global/homes/l/linyaoly/ERA5/reanalysis/omi/'+str(i)+'.csv', header=0)
    ds = ds.iloc[0:365, 1:4]

    pc1 = ds.PC1.values
    pc2 = ds.PC2.values
    initial_amp = np.sqrt(pc1 * pc1 + pc2 * pc2)   
    ds['IniAmp'] = initial_amp

    ############# loop start to evaluate Unet for different lead and memory ###################
    # set lead and memory
    for nmem1 in ["1","5","15","30"]:
        BCC = np.zeros(30)
        RMSE = np.zeros(30)
        amp_err = np.zeros(30)
        pha_err = np.zeros(30)
        for flg in np.arange(1,31): # lead 
            ########### read PCs in prediction and truth ################
            dsp = pd.read_csv(dir+'predicted_UNET_'+flg1+'OMI_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['OMIp1', 'OMIp2'])
            dst = pd.read_csv(dir+'truth_UNET_'+flg1+'OMI_35yr_lead'+str(flg)+'_dailyinput_mem'+str(nmem1)+'d'+str(j)+'.csv', header=None, names=['OMIt1', 'OMIt2'])
            ############## select strong MJO #################
            ds0 = pd.concat([ds, dsp, dst], axis=1)  # Date	PC1	PC2	IniAmp	OMIp1	OMIp2	OMIt1	OMIt2

            strongMJO = ds0[ds0['IniAmp']>=1.0]
            del ds0
            BCC[flg-1], RMSE[flg-1], amp_err[flg-1], pha_err[flg-1] = eval_error(strongMJO)
            del strongMJO
        eval = {'BCC':BCC, 'RMSE':RMSE, 'AmpErr':amp_err, 'PhaErr':pha_err}
        eval = pd.DataFrame(data=eval,index=np.arange(1,31))
        eval.to_csv('./eval/UNET_'+flg1+'OMI_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(i)+'.csv', index_label='lead')
        del eval







