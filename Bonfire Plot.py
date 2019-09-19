#!/usr/bin/env python
# coding: utf-8

# # Plotting PM2.5 On bonfire night using the UO api v1.1

# #### import modules

# In[1]:


import requests
from IPython.display import GeoJSON
import pandas as pd
import io


# #### Retrieve Sensors

# In[2]:


sensor_params = dict(
    sensor_type='PM2.5',
subset_layer='Post code Districts',
subset_shapes='NE1',
)

r = requests.get('http://uoweb3.ncl.ac.uk/api/v1.1/sensors/csv/',sensor_params)

sensor_info = pd.read_csv(io.StringIO(r.text) )
sensor_info


# #### plot location of sensors

# In[3]:


import smopy
import matplotlib.patheffects as pe
import matplotlib.image as image

bbox = (
min(sensor_info['Sensor Centroid Latitude']),
    min(sensor_info['Sensor Centroid Longitude']),
    max(sensor_info['Sensor Centroid Latitude']),
    max(sensor_info['Sensor Centroid Longitude']),
)

map = smopy.Map(bbox, z=17)
ax = map.show_mpl(figsize=(15, 15))
for i,row in sensor_info.iterrows():
    x, y = map.to_pixels(row['Sensor Centroid Latitude'], row['Sensor Centroid Longitude'])
    ax.plot(x, y, 'ob', ms=20, mew=2);
    


# #### setup data parameters

# In[4]:


import datetime

api_date_string_format = "%Y%m%d%H%M%S"

start_time = datetime.datetime(2018,11,5,0)
end_time = datetime.datetime(2018,11,6)

data_params = dict(
    data_variable='PM2.5',
    agg_method='median',
    agg_period='15mins',
    starttime=start_time.strftime(api_date_string_format),
    endtime=end_time.strftime(api_date_string_format) 
)

data_params.update(sensor_params)
data_params


# #### get aggregated data

# In[5]:


r = requests.get('http://uoweb3.ncl.ac.uk/api/v1.1/sensors/data/agg/csv/',data_params)
r


# #### Read into a Pandas DataFrame

# In[6]:


bonfire_data = pd.read_csv(io.StringIO(r.text) )
bonfire_data.head(10)


# #### Plot Data

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30,10))
for sensor_name,sensor_data in  bonfire_data.groupby('Sensor Name'):
    
    datetimes = pd.to_datetime(sensor_data['Timestamp'])
    plt.plot(datetimes,sensor_data['Value'],label=sensor_name)

plt.legend()


# #### Plot Median

# In[8]:


bonfire_data_median = bonfire_data.copy()
bonfire_data_median.index = pd.to_datetime(bonfire_data_median['Timestamp'])
bonfire_data_median = bonfire_data_median.resample('900s').median()

fig, ax = plt.subplots(figsize=(30,10))

plt.plot(bonfire_data_median.index,bonfire_data_median['Value'])

plt.legend()


# ## Clock Plot of Median

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math

idx = pd.date_range(start_time,end_time - datetime.timedelta(minutes=15), freq='900s')
    #raise NameError(idx)
#bonfire_data_median = bonfire_data_median.reindex(idx, fill_value=None)
#bonfire_data_median = bonfire_data_median.where((pd.notnull(bonfire_data_median)), None)
#print(bonfire_data_median)
min_bonfire_pm25,max_bonfire_pm25 = min(bonfire_data_median['Value']),max(bonfire_data_median['Value'])

bon_y_max = math.ceil(max_bonfire_pm25/25)*25

normed = bonfire_data_median['Value']

normed = normed/bon_y_max
max_normed = max(normed)
normed = normed.reindex(idx, fill_value=None)
normed = normed.where((pd.notnull(normed)), None)
print(len(normed))

N = len(normed)
bottom = 2
max_height = 6

theta = (np.linspace(0.0, 2 * np.pi, N, endpoint=False) + (2 * np.pi/N)/2) + np.pi/2


radii = max_height*normed[::-1]
width = (2*np.pi) / N
plt.figure(figsize=(12,12))
ax = plt.subplot(111, polar=True)
ax.set_facecolor('skyblue')
clock_ticks = list(range(23,2,-1) )+[2,1]
clock_ticks = ['0',] + clock_ticks 
clock_ticks = clock_ticks[18:] + clock_ticks[:-6]

minute_d = 2 * np.pi / (24*60)

sunrise = datetime.datetime(2018,11,5,7,0)

sunset = datetime.datetime(2018,11,5,16,24)
minutes_dark = int((datetime.datetime(2018,11,6) - sunset).total_seconds()/60)

sunset_start = int((sunset - datetime.datetime(2018,11,5)).total_seconds()/60)

for i in range(int(minutes_dark/4)):
    minute_start = sunset_start    
    plt.axvspan((minute_d*(i))+ np.pi/2,(minute_d*((i*4)+1))+ np.pi/2, facecolor='#3c4142', alpha=1,ymax=1.1, lw=2)
    

morning_dark = int((sunrise - datetime.datetime(2018,11,5)).total_seconds()/60)

for i in range(int(morning_dark/4)+1):
    plt.axvspan(2 * np.pi - minute_d*(morning_dark-i)+ np.pi/2,2 * np.pi -minute_d*(morning_dark-((i*4)+1))+ np.pi/2,ymax=1.1, facecolor='#3c4142', alpha=1, lw=2)



plt.xticks(ticks=(np.pi/180. * np.linspace(0,  360, 24, endpoint=False)), labels=clock_ticks)

bars = ax.bar(theta, radii, width=width, bottom=bottom)

defra = bottom+(((25)/(max_bonfire_pm25))*max_height)

y_ticks = [0,bottom,defra,8]

for i in range(720):
    
    plt.hlines(defra,(i*2)*minute_d,((i*2)+1)*minute_d, color='blue', linestyle='-',lw=2)
    
    
cmap = plt.get_cmap('autumn_r')
for bar in bars:
    bar.set_color(cmap(bar.get_height()/(max_normed* max_height)))
plt.title('pm2.5 Bonfire Night 2018')
plt.yticks(ticks=y_ticks,labels=[None,'0ugm-3','25ugm-3\n (Defra annual mean limit)',str(bon_y_max)+'ugm-3'],color='black',path_effects=[pe.withStroke(linewidth=4, foreground="white")])
plt.ylim((0,8))




plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




