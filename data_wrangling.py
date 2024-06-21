import os
import pandas as pd
import numpy as np

def read_weather_files():
    syd_df =[]
    melb_df = []
    bris_df = []
    adelaide_df = []
    for f in os.listdir('/course/data/a2/weather'):
        if "weather" in f:
            if "sydney" in f:
                syd_df = clean_weather_files("NSW1")
            elif "melbourne" in f:
                melb_df = clean_weather_files("VIC1")
            elif "adelaide" in f:
                adelaide_df = clean_weather_files("SA1")
            elif "brisbane" in f:
                bris_df = clean_weather_files("QLD1")        
       
    return syd_df, melb_df, bris_df, adelaide_df 

def clean_price_demand_file():
    price_demand_df = pd.read_csv('/course/data/a2/weather/price_demand_data.csv')

    date_time = price_demand_df['SETTLEMENTDATE'].str.split(' ', expand= True)
    price_demand_df['Date'] = date_time[0]
    price_demand_df['Time'] = date_time[1]
    price_demand_df = price_demand_df.drop(columns= ['SETTLEMENTDATE'])
    price_demand_df['Date'] = pd.to_datetime(price_demand_df['Date'])
    price_demand_df['Time'] = price_demand_df['Time'].str[:-3]

    return price_demand_df

def clean_weather_files():

    weather_df = pd.read_csv('/course/data/a2/weather/weather_melbourne.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_df['REGION'] = 'VIC1'
    # Detecting numbers 
    cnt=0
    for row in weather_df['3pm wind speed (km/h)']: #should loop for other float columns too
        try:
            int(row)
        except ValueError:
            weather_df.loc[cnt, '3pm wind speed (km/h)'] = np.nan
        cnt+=1
    weather_df['3pm wind speed (km/h)'] = weather_df['3pm wind speed (km/h)'].apply(pd.to_numeric)
    
    return weather_df
