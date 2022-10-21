import numpy as np 
import pandas as pd 
import xarray as xr
import matplotlib.pyplot as plt
import glob
import os 

flg1 = 'u200u850olrprepv200T200_'
# set year start and end
ystat = 2017
yend = 2020

###################################### get av ###################################################
###################################### get av ###################################################
###################################### get av ###################################################
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20,12))
ax1 = fig.add_axes([0.1, 0.55, 0.33, 0.33])
ax2 = fig.add_axes([0.5, 0.55, 0.33, 0.33])
ax3 = fig.add_axes([0.1, 0.15, 0.33, 0.33])
ax4 = fig.add_axes([0.5, 0.15, 0.33, 0.33])
plt.rcParams.update({'font.size': 20})

for nmem1 in ["1","30"]:
    ds = pd.read_csv('./eval/UNET_'+flg1+'RMMERA5_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(ystat)+'to'+str(yend)+'av.csv')
    ax1.plot(np.arange(1,31),ds.BCC.values,'o-', label=nmem1+'-d mem')
    ax2.plot(np.arange(1,31),ds.RMSE.values,'o-')
    ax3.plot(np.arange(1,31),ds.AmpErr.values,'o-')
    ax4.plot(np.arange(1,31),ds.PhaErr.values,'o-')

ax1.set_ylim([0.0, 1.0])
ax1.set_title('BCC')
ax1.grid(visible=True)
ax1.plot(np.arange(1,31),np.ones(30)*0.5,'k--')
# ax1.set_xlabel('lead')
ax1.legend()
ax2.set_title('RMSE')
ax2.set_ylim([0.4, 1.6])
ax2.grid(visible=True)
# ax2.set_xlabel('lead')
ax3.set_xlabel('lead')
ax3.set_title('Amplitude error')
ax3.set_ylim([-1.0, 0.0])
ax3.grid(visible=True)
ax4.set_xlabel('lead')
ax4.set_title('Phase error')
ax4.plot(np.arange(1,31),np.zeros(30),'k--')
ax4.set_ylim([-12.5, 10])

plt.rcParams.update({'font.size': 20})
plt.show()
plt.savefig('./eval/UNET_'+flg1+'RMMERA5_35yr_dailyinput_'+str(ystat)+'to'+str(yend)+'av.png', bbox_inches='tight', dpi=300, transparent=False)
plt.close()

###################################### get spread in years ###################################################
###################################### get spread in years ###################################################
###################################### get spread in years ###################################################
nmem1 = '1'

fig = plt.figure(figsize=(20,12))
ax1 = fig.add_axes([0.1, 0.55, 0.33, 0.33])
ax2 = fig.add_axes([0.5, 0.55, 0.33, 0.33])
ax3 = fig.add_axes([0.1, 0.15, 0.33, 0.33])
ax4 = fig.add_axes([0.5, 0.15, 0.33, 0.33])

for yn in np.arange(ystat,yend):  # , "30"]:
    ds = pd.read_csv('./eval/UNET_'+flg1+'RMMERA5_35yr_dailyinput_mem'+str(nmem1)+'d_'+str(yn)+'.csv')
    ax1.plot(np.arange(1,31),ds.BCC.values,'o-', label=str(yn))
    ax2.plot(np.arange(1,31),ds.RMSE.values,'o-')
    ax3.plot(np.arange(1,31),ds.AmpErr.values,'o-')
    ax4.plot(np.arange(1,31),ds.PhaErr.values,'o-')

ax1.set_ylim([0.0, 1.0])
ax1.set_title('BCC')
ax1.grid(visible=True)
ax1.plot(np.arange(1,31),np.ones(30)*0.5,'k--')
# ax1.set_xlabel('lead')
ax1.legend()
ax2.set_title('RMSE')
ax2.set_ylim([0.4, 1.6])
ax2.grid(visible=True)
# ax2.set_xlabel('lead')
ax3.set_xlabel('lead')
ax3.set_title('Amplitude error')
ax3.set_ylim([-1.0, 0.0])
ax3.grid(visible=True)
ax4.set_xlabel('lead')
ax4.set_title('Phase error')
ax4.plot(np.arange(1,31),np.zeros(30),'k--')
ax4.set_ylim([-12.5, 10])

plt.rcParams.update({'font.size': 20})
plt.show()
plt.savefig('./eval/UNET_'+flg1+'RMMERA5_35yr_dailyinput_'+nmem1+'dmem_'+str(ystat)+'to'+str(yend)+'sp.png', bbox_inches='tight', dpi=300, transparent=False)
plt.close()