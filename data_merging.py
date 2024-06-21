import random 
import pandas as pd
import numpy as np

def merge_data(price_demand, weather):

    price_demand = price_demand[price_demand.REGION == "VIC1"]

    weather['9am Total Demand'], weather['9am Price Surge'] = get_column(price_demand, weather, "09")
    weather['3pm Total Demand'], weather['3pm Price Surge']  = get_column(price_demand, weather, "15")
    weather['month'] = weather['Date'].dt.month 
    
    jan, feb, march, april, may, jun, jul, aug, sep, octuber, nov, dec = \
        weather[weather.month == 1], weather[weather.month == 2], weather[weather.month == 3], \
        weather[weather.month == 4], weather[weather.month == 5], weather[weather.month == 6], \
        weather[weather.month == 7], weather[weather.month == 8], weather[weather.month == 9],\
        weather[weather.month == 10], weather[weather.month == 11], weather[weather.month == 12]


    aut, win, spr, smr = pd.concat([march, april, may]).drop('month', axis=1), pd.concat([jun, jul, aug]).drop('month', axis=1), \
        pd.concat([sep, octuber, nov]).drop('month', axis=1),  pd.concat([dec, jan, feb]).drop('month', axis=1)
   
    return aut, win, spr, smr


def get_column(price_demand, weather, hour):

    total_demand_column = []
    price_surge_column = []
    j = 0
    i = 0 
    count = 0 
    total_demand = 0
    false_price_surge_count = 0
    true_price_surge_count = 0


    for w in weather['Date']:
        total_demand = 0

        
        price_demand_df = price_demand[price_demand.REGION == weather['REGION'][i]]
        price_demand_df = price_demand_df[price_demand_df.Date == w]
        price_demand_df = price_demand_df.reset_index(drop = True)

        for time in price_demand_df['Time']:

            if str(time[0:2]) == hour:
                total_demand += price_demand_df['TOTALDEMAND'][j]
                count += 1
                if price_demand_df['PRICESURGE'][j] == True:
                    true_price_surge_count += 1
                elif price_demand_df['PRICESURGE'][j] == False:
                    false_price_surge_count += 1
                
            j += 1
        i += 1
        j = 0

        if (count != 0):
            total_demand_column.append(total_demand/count)
        else:
            total_demand_column.append(0)
        count = 0

        if (true_price_surge_count > false_price_surge_count):
            price_surge_column.append(True)
        elif (true_price_surge_count < false_price_surge_count):
            price_surge_column.append(False)
        else:
            price_surge_column.append(np.nan)

    return total_demand_column, price_surge_column



