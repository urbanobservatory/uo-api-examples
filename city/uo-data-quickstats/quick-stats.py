#!/usr/bin/env python
# coding: utf-8

# # Urban Observatory Sensor Quick Stats

# In[4]:


#%matplotlib inline

# imports
import pandas as pd
from pandas.tseries.offsets import *
import glob

import datetime
from datetime import timedelta

from matplotlib import pyplot as plt

# missing data
import missingno as msno


# ## Read in the data file
# Get data from Urban Observatory __[download](http://uoweb1.ncl.ac.uk/download/)__ website

# In[5]:


# working folder
folder = 'baseline/'

# data file
source = folder + 'data.csv'

# sensor description file (Epoch is sampling frequency)
# Sensor Name | Location Name | Location (WKT) | Ground Height Above Sea Level | Sensor Height Above Ground | Broker Name | Third Party | Sensor Centroid Longitude Sensor Centroid Latitude | DeploymentStart | DeploymentEnd | Epoch
# PER_AIRMON_MESH1971150 | Coast Road Corner House | POINT (-1.58378809690475 54.9911393751596) | 45.54999924 | 2 | Air Monitors | FALSE | -1.583788097 | 54.99113938 | 25/06/2018 | | 900
sensors = folder + 'sensors.csv'

rawdata=pd.read_csv(source,sep=',',index_col=False, header=0)
rawdata = rawdata.drop_duplicates()
#rawdata = rawdata[rawdata['Variable'] == 'NO2' ]
sensordata = pd.read_csv(sensors,sep=',',index_col=False, header=0)
#sensordata = pd.concat([pd.read_csv(f) for f in glob.glob(sensors)], ignore_index = True)

# get deplyment times and fill in current time if end missing
sensordata['DeploymentEnd'].fillna(datetime.datetime.now().date(),inplace=True)
sensordata['DeploymentStart'] = pd.to_datetime(sensordata['DeploymentStart'])
sensordata['DeploymentEnd'] = pd.to_datetime(sensordata['DeploymentEnd'])

# convert to datetime
rawdata['Timestamp'] = pd.to_datetime(rawdata['Timestamp'])
rawdata = rawdata.set_index('Timestamp')
#includes times that doesn't exit
#rawdata = rawdata.tz_localize('utc').reset_index()
sensordata


# ## Visualise Sensor Locations

# In[6]:


import smopy
import matplotlib.patheffects as pe

bbox = (
    min(sensordata['Sensor Centroid Latitude']),
    min(sensordata['Sensor Centroid Longitude']),
    max(sensordata['Sensor Centroid Latitude']),
    max(sensordata['Sensor Centroid Longitude']),
)

map = smopy.Map(bbox, z=17)
ax = map.show_mpl(figsize=(15, 15))
for i,row in sensordata.iterrows():
    x, y = map.to_pixels(row['Sensor Centroid Latitude'], row['Sensor Centroid Longitude'])
    ax.plot(x, y, 'ob', ms=20, mew=2);
    


# ## TODO: Filter out deployment times

# ## Data Quality assessment

# In[8]:


# remove data that is outside the deployment times (in case sensor had been taken to GOLD_NODE turing the deployment)
cleandata = pd.DataFrame()
for index, row in sensordata.iterrows():
    mask = (((rawdata.index >= row['DeploymentStart']) & (rawdata.index <= row['DeploymentEnd'])))
    sensor_set = rawdata.loc[(rawdata['Sensor Name'] == row['Sensor Name']) & (mask)]
    cleandata = cleandata.append(sensor_set)

cleandata = cleandata.reset_index()
cleandata.head()


# ### Number of Flagged as Suspect Readings

# In[9]:


data = cleandata.copy()

# number of flagged readings
flagged = data[data['Flagged as Suspect Reading'] == True].groupby(['Sensor Name',  'Variable'])['Flagged as Suspect Reading'].value_counts()
flagged.to_csv(folder+'flagged_as_suspect_reading.csv', encoding='utf-8',header=True)
flagged


# ### Min and Max 
# For each sensor

# In[10]:


# min max datetimes for all sensors in the list
data = cleandata.copy()
data = pd.pivot_table(cleandata, values = 'Value', index=['Sensor Name','Timestamp'], columns = 'Variable').reset_index()
minmax = data.groupby('Sensor Name')['Timestamp'].agg(['min','max']).reset_index()
minmax.to_csv(folder+'min_max.csv', encoding='utf-8',header=True,index=False)
minmax


# ### Missing Data
# 
# If sensors get relocated the missing data gets messed up

# In[11]:


#Data quality assessment
data = pd.pivot_table(cleandata, values = 'Value', index=['Sensor Name','Timestamp'], columns = 'Variable').reset_index()

#min max of all sensors
minmax = data.groupby('Sensor Name')['Timestamp'].agg(['min','max']).reset_index()

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)

datagaps = pd.DataFrame()
datagaps['Sensor Name'] = data['Sensor Name'].unique()

#finding gaps in data
for index, row in minmax.iterrows():
    sensor_set = data.loc[data['Sensor Name'] == row['Sensor Name']]
    sensor_set = sensor_set.drop(columns=['Sensor Name'])
    # TODO: hard coded frequency, should be actually the sampling freq (now set all to 1min from 16 November 2018 - http://uoweb3.ncl.ac.uk/tools/note/8164195/) 
    idx_ref = pd.DatetimeIndex(start=row['min'], end=row['max'],freq='15min')                     
    gaps = idx_ref[~idx_ref.isin(sensor_set)]
    datagaps.loc[datagaps.index[datagaps['Sensor Name'] == row['Sensor Name']], 'Number of Missing Datapoints'] = int(len(gaps))
    dates = pd.DataFrame()
    dates['Timestamp'] = pd.date_range(start=row['min'], end=row['max'], freq='15min')
    missing_data = dates.merge(sensor_set, how='outer')
    missing_data = missing_data.set_index('Timestamp')
    #sensor_set.set_index('Timestamp')
    #sensor_set.plot(x = 'Timestamp', y='Value')
    #g = sns.FacetGrid(sensor_set, col='Variable', col_wrap=2)
    #g = g.map_dataframe(dateplot, 'Timestamp', 'Value')
    #print(missing_data.head())
    #print(row['min'], row['max'])
    try:
        ax = msno.matrix(missing_data,labels='Timestamp',freq='W')
        fig = ax.get_figure()
        fig.suptitle(row['Sensor Name'], fontsize=16)
        fig.set_size_inches(25,15)
        #fig.savefig(row['Sensor Name']+".png",bbox_inches='tight')
        fig.savefig(folder+row['Sensor Name']+'-missing_data.png')
    except Exception as e:
        print("Error processing " + row['Sensor Name'])
        print(e)
    
datagaps.to_csv(folder+'missing_data.csv', encoding='utf-8',header=True,index=False)
    


# ### Cleanup

# In[13]:


# remove suspect readings (also consider some additional outliers)
data = cleandata.copy()
#data = pd.pivot_table(rawdata, values = 'Value', index=['Sensor Name','Timestamp'], columns = 'Variable').reset_index()
#data = data.loc[data['NO2'] > 200]
cdata = data.drop(data[data['Flagged as Suspect Reading'] == True].index)
cdata.head()


# ## Compute Quick Stats

# ### Overall Means

# In[25]:


def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}

#overall means
data = cdata.copy()
data = data[['Sensor Name','Timestamp','Variable','Value','Units']]
minmax = data.groupby('Sensor Name')['Timestamp'].agg(['min','max']).reset_index()
g1 = data.groupby( ['Sensor Name', 'Variable','Units'] ).agg(['mean', 'count','max','min']).reset_index()

for index, row in minmax.iterrows():
    days = (row['max'] - row['min']).days
    g1.loc[g1.index[g1['Sensor Name'] == row['Sensor Name']], 'Timeframe'] = row['min'].strftime('%d/%m/%Y %H:%M:%S') +" - "+row['max'].strftime('%d/%m/%Y %H:%M:%S')
    # should use 60 second as Epoch because http://uoweb3.ncl.ac.uk/tools/note/8164195/
    # however different sensors have different sampling frequency
    epoch = sensordata.loc[sensordata.index[sensordata['Sensor Name'] == row['Sensor Name']]]['Epoch'].astype(int)
    g1.loc[g1.index[g1['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = (days*86400)/900
g1['DataCompleteness'] = (g1['Value']['count']*100)/g1['MaxReadings']
g1.to_csv(folder+'overall_means.csv', encoding='utf-8',header=True,index=False)
g1


# ### Weekly means
# 
# Week starts with Monday date

# In[27]:


#weekly means
data = cdata.copy()
data = data[['Sensor Name','Timestamp','Variable','Value','Units']]


g2 = data.groupby(['Sensor Name', 'Variable','Units']).resample('W-Mon', closed='left', label='left', on='Timestamp').apply(get_stats).reset_index().sort_values('Timestamp')
g2[['min', 'max','count','mean']] = g2['Value'].apply(pd.Series)
g2 = g2.drop(columns=['Value'])

for index, row in sensordata.iterrows():
    # should use 60 second as Epoch because http://uoweb3.ncl.ac.uk/tools/note/8164195/
    # however different sensors have different sampling frequency
    g2.loc[g2.index[g2['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = (7*86400)/900

g2['DataCompleteness'] = (g2['count']*100)/g2['MaxReadings']

g2.to_csv(folder+'weekly_means.csv', encoding='utf-8',header=True,index=False)
g2.head()


# ### Weekday (Mon - Friday) Means

# In[29]:


#weekday means
data = cdata.copy()
data = data[['Sensor Name','Timestamp','Variable','Value','Units']]
data['weekday'] = data['Timestamp'].apply(lambda x: x.weekday())
g3 = data[data['weekday'] < 5 ]
g3 = g3.drop('weekday', 1)
g3 = g3.groupby(['Sensor Name', 'Variable']).resample('W-Mon', closed='left', label='left', on='Timestamp').apply(get_stats).reset_index().sort_values('Timestamp')
g3[['min', 'max','count','mean']] = g3['Value'].apply(pd.Series)
g3 = g3.drop(columns=['Value'])

for index, row in sensordata.iterrows():
    g3.loc[g3.index[g3['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = (5*86400)/900

g3['DataCompleteness'] = (g3['count']*100)/g3['MaxReadings']

g3.to_csv(folder+'weekday_means.csv', encoding='utf-8',header=True,index=False)
g3.head()


# ### Weekend (Sat - Sun) Means

# In[30]:


#weekend means
data = cdata.copy()
data = data[['Sensor Name','Timestamp','Variable','Value','Units']]
data['weekday'] = data['Timestamp'].apply(lambda x: x.weekday())

g4 = data[data['weekday'] >= 5 ]
g4 = g4.drop('weekday', 1)


g4 = g4.groupby(["Sensor Name", "Variable"]).resample('W-Sat', closed='left', label='left', on='Timestamp').apply(get_stats).reset_index().sort_values('Timestamp')
g4[['min', 'max','count','mean']] = g4['Value'].apply(pd.Series)
g4 = g4.drop(columns=['Value'])

for index, row in sensordata.iterrows():
    # should use 60 second as Epoch because http://uoweb3.ncl.ac.uk/tools/note/8164195/
    # however different sensors have different sampling frequency
    g4.loc[g4.index[g4['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = (2*86400)/900

g4['DataCompleteness'] = (g4['count']*100)/g4['MaxReadings']


g4.to_csv(folder+'results/weekend_means.csv', encoding='utf-8',header=True,index=False)
g4.head()


# ### Hourly Means

# In[33]:


data = cdata.copy()
g5 = data[['Sensor Name','Timestamp','Variable','Value','Units']]


g5 = g5.groupby(["Sensor Name", "Variable"]).resample('H', closed='left', label='left', on='Timestamp').apply(get_stats).reset_index().sort_values('Timestamp')
g5[['min', 'max','count','mean']] = g5['Value'].apply(pd.Series)
g5 = g5.drop(columns=['Value'])

for index, row in sensordata.iterrows():
    # should use 60 second as Epoch because http://uoweb3.ncl.ac.uk/tools/note/8164195/
    # however different sensors have different sampling frequency
    g5.loc[g5.index[g5['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = 3600/900

g5['DataCompleteness'] = (g5['count']*100)/g5['MaxReadings']

# AQ index for NO2
'''
Reading,Index,Band,BandLow,BandHigh
NO2,1,Low,0,67
NO2,2,Low,68,134
NO2,3,Low,135,200
NO2,4,Moderate,201,267
NO2,5,Moderate,268,334
NO2,6,Moderate,335,400
NO2,7,High,401,467
NO2,8,High,468,534
NO2,9,High,535,600
NO2,10,Very High,601
'''
# aq_index=pd.read_csv("aq_index.csv",sep=',',index_col=False, header=0)
#for index,row in aq_index.iterrows():
#    g5.loc[g5.index[(g5['mean'] >= row['BandLow']) & (g5['mean'] <= row['BandHigh'])], 'Index'] = row['Index']

g5.to_csv(folder+'hourly_means.csv', encoding='utf-8',header=True,index=False)
g5.head()


# ### Hourly Means for sensors for the whole period

# In[35]:


data = cdata.copy()
g51 = data[['Sensor Name','Timestamp','Variable','Value','Units']]
g51['hourofday'] = g51['Timestamp'].apply(lambda x: x.hour)
g51 = g51.groupby( ['Sensor Name', 'Variable','hourofday'] ).agg(['mean','count','max','min']).reset_index()
g51[['mean', 'count','max','min']] = g51['Value'].apply(pd.Series)
g51 = g51.drop(columns=['Value'])

for index, row in minmax.iterrows():
    days = (row['max'] - row['min']).days
    g51.loc[g51.index[g51['Sensor Name'] == row['Sensor Name']], 'Timeframe'] = row['min'].strftime('%d/%m/%Y %H:%M:%S') +" - "+row['max'].strftime('%d/%m/%Y %H:%M:%S')
    epoch = sensordata.loc[sensordata.index[sensordata['Sensor Name'] == row['Sensor Name']]]['Epoch']
    # from 16 November 2018 all sensors are 1min freq http://uoweb3.ncl.ac.uk/tools/note/8164195/
    # so it should be 60 readings in an hour
    g51.loc[g51.index[g51['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = days*4
g51['DataCompleteness'] = (g51['count']*100)/g51['MaxReadings']
#g1.to_csv(folder+'results/overall_means.csv', encoding='utf-8',header=True,index=False)
g51.to_csv(folder+'overall_hourly_means.csv', encoding='utf-8',header=True,index=False)
g51


# ### Hourly means over the period

# In[171]:


data = cdata.copy()
data = data[['Sensor Name','Timestamp','Variable','Value','Units']]
data['hourofday'] = data['Timestamp'].apply(lambda x: x.hour)

data['week_start'] = data['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)


g6 = data.groupby( ['Sensor Name', 'Variable','Units','hourofday','week_start'] ).agg(['mean','count','max','min']).reset_index()
g6[['mean', 'count','max','min']] = g6['Value'].apply(pd.Series)
g6 = g6.drop(columns=['Value'])

for index, row in sensordata.iterrows():
    g6.loc[g6.index[g6['Sensor Name'] == row['Sensor Name']], 'MaxReadings'] = (7*3600)/900

g6['DataCompleteness'] = (g6['count']*100)/g6['MaxReadings']
#g6.to_csv(folder+'results/hour_means_byweek.csv', encoding='utf-8',header=True,index=False)

#sns.catplot(x='hourofday', y='mean',col='week_start', data=g6,kind='bar',height=10, aspect=1);
#g = sns.FacetGrid(g6, col="week_start", y='mean', height=4, aspect=.5)
#g = g.map(plt.hist, "hoursofday", bins=bins)

grouped = g6.groupby('Sensor Name')

for name, group in grouped:
    #g = sns.catplot(x='hourofday', y='mean',col='week_start', data=group,kind='bar')
    group['week_start'] = group['week_start'].dt.strftime('%Y-%m-%d')
    g = sns.FacetGrid(group, col="week_start")
    g = g.map(plt.bar, "hourofday", "mean")
    g.fig.savefig(folder+name+"byweekly-hourly.png")


# ## Compile Report

# In[105]:


import xlsxwriter
import glob
import csv
import os

workbook = xlsxwriter.Workbook(folder+'results/compiled.xlsx') 
for filename in glob.glob(folder+"results/*.csv"):
    (f_path, f_name) = os.path.split(filename)
    (f_short_name, f_extension) = os.path.splitext(f_name)
    ws = workbook.add_worksheet(f_short_name)
    spamReader = csv.reader(open(filename, 'rt'), delimiter=',',quotechar='"')
    row_count = 0
    print(f_short_name)
    for row in spamReader:
        for col in range(len(row)):
            ws.write(row_count,col,row[col])
        row_count +=1

workbook.close()