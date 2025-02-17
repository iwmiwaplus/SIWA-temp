# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:18:51 2020

@author: esa
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import glob

def open_nc(nc,timechunk=1,chunksize=1000):
    dts=xr.open_dataset(nc)
    key=list(dts.keys())[0]
    var=dts[key].chunk({"time": timechunk, "latitude": chunksize, "longitude": chunksize}) #.ffill("time")
    return var,key

basin = 'Menasagi'
MAIN_FOLDER = r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Menasagi'
p_in = os.path.join(MAIN_FOLDER,'%s_P_CHIRPS.nc'%(basin)) # Monthly Precipitation
e_in = os.path.join(MAIN_FOLDER,'%s_ETa_SSEBop.nc'%(basin)) # Monthly Actual Evapotranspiration
i_in = os.path.join(MAIN_FOLDER,'i_monthly.nc') # Monthly Interception
nrd_in = os.path.join(MAIN_FOLDER,'nRD_monthly.nc') # Monthly Number of Rainy days

bf_in = os.path.join(MAIN_FOLDER,'bf_monthly.nc')
d_perco_in = os.path.join(MAIN_FOLDER,'d_perco_monthly.nc')
d_sro_in = os.path.join(MAIN_FOLDER,'d_sro_monthly.nc')
et_incr_in = os.path.join(MAIN_FOLDER,'etincr_monthly.nc')
et_rain_in = os.path.join(MAIN_FOLDER,'etrain_monthly.nc')
perco_in = os.path.join(MAIN_FOLDER,'perco_monthly.nc')
sm_in = os.path.join(MAIN_FOLDER,'sm_monthly.nc')
sro_in = os.path.join(MAIN_FOLDER,'sro_monthly.nc')
supply_in = os.path.join(MAIN_FOLDER,'supply_monthly.nc')
tf_in = os.path.join(MAIN_FOLDER,'tf_monthly.nc')
gw_in = os.path.join(MAIN_FOLDER,'gw_monthly.nc')

start_date='2010-06-01' 
end_date = '2018-05-31'

dates = pd.date_range(start_date,end_date,freq = 'MS')


Pt,_=open_nc(p_in)
P_ts = Pt.mean(dim=['latitude', 'longitude'])
Et,_=open_nc(e_in)
Et_ts = Et.mean(dim=['latitude', 'longitude'])
I,_ = open_nc(i_in)
I_ts = I.mean(dim=['latitude', 'longitude'])
Bf,_ = open_nc(bf_in)
BF_ts = Bf.mean(dim=['latitude', 'longitude'])
d_perco,_ = open_nc(d_perco_in)
d_perco_ts = d_perco.mean(dim=['latitude', 'longitude'])
d_sro,_ = open_nc(d_sro_in)
d_sro_ts = d_sro.mean(dim=['latitude', 'longitude'])
ET_incr,_ = open_nc(et_incr_in)
ET_incr_ts = ET_incr.mean(dim=['latitude', 'longitude'])
ET_rain,_ = open_nc(et_rain_in)
ET_rain_ts = ET_rain.mean(dim=['latitude', 'longitude'])
Perco,_ = open_nc(perco_in)
Perco_ts = Perco.mean(dim=['latitude', 'longitude'])
SM,_ = open_nc(sm_in)
SM_ts = SM.mean(dim=['latitude', 'longitude'])
SRO,_ = open_nc(sro_in)
SRO_ts = SRO.mean(dim=['latitude', 'longitude'])
Supply,_ = open_nc(supply_in)
Supply_ts = Supply.mean(dim=['latitude', 'longitude'])
TF,_=open_nc(tf_in)
TF_ts = TF.mean(dim=['latitude', 'longitude'])
GW,_=open_nc(gw_in)
GW_ts = GW.mean(dim=['latitude', 'longitude'])

Dataset = pd.DataFrame(index = dates, data = P_ts, columns=['P'])
Dataset['ET'] = Et_ts
Dataset['Interc'] = I_ts
Dataset['BF'] = BF_ts
Dataset['d_perco'] = d_perco_ts
Dataset['d_sro'] = d_sro_ts
Dataset['ET_incr'] = ET_incr_ts
Dataset['ET_rain'] = ET_rain_ts
Dataset['Perco'] = Perco_ts
Dataset['SM'] = SM_ts
Dataset['SRO'] = SRO_ts
Dataset['Supply'] = Supply_ts
Dataset['TF'] = TF_ts
Dataset['GW'] = GW_ts

f_name = os.path.join(MAIN_FOLDER,'results.csv')
Dataset.to_csv(f_name)


#%%
# check discharge
import os
os.chdir(r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Scripts')
from WB_fuctions import SW_return_abstraction
from WAsheets import calculate_flux as cf
from WAsheets import hydroloop as hl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

MAIN_FOLDER = r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Menasagi'

# inputs
mask = r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Data\Boundaries\Menasagi.tif'
aeisw = r"E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Data\gmia_v5_aeisw_pct_aei.tif"
population = r"E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Data\Pop_Menasagi.tif"


yearly_nc=os.path.join(MAIN_FOLDER,'Menasagi_LU_WA.nc')
sample_nc=os.path.join(MAIN_FOLDER,'Menasagi_P_CHIRPS.nc')
supply = os.path.join(MAIN_FOLDER,'supply_monthly.nc')
et_incr = os.path.join(MAIN_FOLDER,'etincr_monthly.nc')
sroincr = os.path.join(MAIN_FOLDER,'d_sro_monthly.nc')
percincr = os.path.join(MAIN_FOLDER,'d_perco_monthly.nc')

# compute return flow to surface water and surface water abstraction
SW_return_abstraction(MAIN_FOLDER, mask, aeisw, population, 
                      yearly_nc, sample_nc, supply, et_incr, sroincr, percincr)

basin_data = {}
for key in ['sro_monthly','return_sw','bf_monthly','total_sw_supply']:
    output=os.path.join(MAIN_FOLDER,
                        '{0}.csv'.format(key))
    df=cf.calc_flux_per_basin(os.path.join(MAIN_FOLDER, '%s.nc' %(key)),
                           mask,
                           output=output)
    basin_data[key]=df
## calculate sw discharge and dS from pixel-based model results
output=os.path.join(MAIN_FOLDER,
                    '{0}.csv')
discharge,dS_sw=hl.calc_sw_from_wp(basin_data['sro_monthly'],
                                   basin_data['return_sw'],
                                   basin_data['bf_monthly'],
                                   basin_data['total_sw_supply'],
                                   inflow=None,
                                   output=output,
                                   outflow=True, #not endorheic basin
                                   )

data_sim=r"E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Menasagi\discharge.csv"
data_insitu = r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Menasagi\In-situ\discharge_Menasagi.csv'

ts_sim = pd.read_csv(data_sim,sep=';',index_col=0)
ts_sim=ts_sim/1e3 # MCM

in_situ_data = pd.read_csv(data_insitu, sep=';', header = 1)
start_df = in_situ_data['datetime'][0]
end_df = in_situ_data['datetime'][len(in_situ_data['datetime'])-1]
dates = pd.date_range(start_df[:10],end_df[:10], freq='MS')
Q_dates = np.array([datetime.date(d.year, d.month, d.day) for d in dates])
in_situ_df = pd.DataFrame(data = list(in_situ_data['data']), index=Q_dates, columns=['In-situ'])

Data = pd.DataFrame(data = ts_sim.values, index = ts_sim.index, columns = ['Simulated Q'])
Data.index = pd.to_datetime(Data.index)
Data['In-situ Q'] = pd.DataFrame(data = list(in_situ_data['data']), index=Q_dates, columns=['In-situ'])

ax=Data.plot()
ax.set_title('Comparison in-situ and simulated discharge')
ax.set_ylabel('km3/month')
fig = ax.get_figure()
fig.savefig(r'E:\AAA_Karnataka\Training_3-5Aug\DAY2_WB\Menasagi\comparison_discharge.png', dpi = 300)




