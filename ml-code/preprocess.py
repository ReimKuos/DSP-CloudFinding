import pandas as pd
import numpy as np
from json import loads, dump
from datetime import datetime as dt, timedelta
import pytz as tz
from astral.sun import sun
from astral import LocationInfo
from fmiopendata.wfs import download_stored_query
from time import sleep

def read_cloud_data(file):
    '''Reads the cloud observation data into a dataframe'''

    df = pd.read_json(file)
    df = df['observation'].to_json(orient='records')
    df = pd.read_json(df)
    df = df[['title', 'start', 'city','coordinates', 'showiness']]
    return df

def format_time(row):
    '''Creates a datetime object from the given datetime string, changes
    the timezone to UTC, and rounds the time to the nearest 10 minutes'''

    dt_str = row['observation_time']

    format = '%Y-%m-%d %H:%M:%S'
    helsinki_tz = tz.timezone('Europe/Helsinki')
    utc_tz = tz.timezone('UTC')

    dt_obj = dt.strptime(dt_str, format)
    dt_obj = helsinki_tz.localize(dt_obj)
    dt_obj = dt_obj.astimezone(utc_tz)

    minutes_past_last_ten = dt_obj.minute % 10
    adjustment = 10 - minutes_past_last_ten
    dt_obj = dt_obj + timedelta(minutes=adjustment)
    return dt_obj

def time_from_sunset(row):
    '''Calculates the time in hours from or to the nearest sunset for the given
    location and datetime. If the sunset has already happened, time is positive,
    otherwise it is negative. If the sun doesn't set at given location time from
    or to the midnight is calculated instead.'''

    lat = row['latitude']
    lon = row['longitude']
    time = row['observation_time']

    loc = LocationInfo(latitude=lat, longitude=lon)
    prev_day = time - timedelta(days=1)
    next_day = time + timedelta(days=1)

    try:
        s1 = sun(loc.observer, date=time.date())
        s2 = sun(loc.observer, date=prev_day.date())
        sunset1 = s1['sunset']
        sunset2 = s2['sunset']
        sunset1 = sunset1.astimezone(time.tzinfo)
        sunset2 = sunset2.astimezone(time.tzinfo)

        if abs(time-sunset1) < abs(time-sunset2):
            sunset = sunset1
        else:
            sunset = sunset2

        return round((time - sunset).total_seconds() / 3600.0, 3)
    except:
        midnight1 = time.replace(hour=0, minute=0, second=0)
        midnight2 = next_day.replace(hour=0, minute=0, second=0)

        if abs(time-midnight1) < abs(time-midnight2):
            midnight = midnight1
        else:
            midnight = midnight2

        return round((time - midnight).total_seconds() / 3600.0, 3)

def get_dist(origin, point):
    '''Returns the distance between the given coordinate
    points by using the haversine formula'''

    # Transform the coordinates into radians
    origin = tuple(map(np.radians, map(float, origin)))
    point = tuple(map(np.radians, map(float, point)))

    # Apply the haversine formula
    r = 6371
    a = (np.sin((point[0]-origin[0])/2))**2
    b = np.cos(origin[0])*np.cos(point[0])
    c = (np.sin((point[1]-origin[1])/2))**2
    return 2*r*np.arcsin(np.sqrt(a+b*c))

def nearest_stations(stations, location):
    '''Returns the names of the observations station
    that are nearest to the given location'''

    distances = []
    for station, data in stations.items():
        lat = data['latitude']
        lon = data['longitude']
        dist = get_dist((lat, lon), location)
        distances.append((station, dist))

    distances = sorted(distances, key=lambda x: x[1])

    return [x[0] for x in distances], [x[1] for x in distances]

def get_weather_obs(row):
    '''Returns the historical weather observation data 
    from the FMI database'''

    sleep(1) # Limits the amount of requests/second to comply with the FMI open data requirements
    start = row['observation_time']
    lat, lon = row['latitude'], row['longitude']

    bbox = [lon-0.25, lat-0.25, lon+0.25, lat+0.25] # The region from which to look for weather observations

    date_format = '%Y-%m-%dT%H:%M:%SZ'
    end = start + timedelta(minutes=0.1)
    start = start.strftime(date_format)
    end = end.strftime(date_format)

    obs = download_stored_query('fmi::observations::weather::multipointcoverage', 
                                    args=['bbox=' + ','.join(map(str,bbox)), 
                                          'starttime=' + start, 
                                          'endtime=' + end])
    
    time = dt.strptime(start, date_format)
    stations, distances = nearest_stations(obs.location_metadata, (lat, lon))

    measurements = {'temps': [],
                  'dew_temps': [],
                  'hums': [],
                  'pressures': [],
                  'c_covers': []}

    for station in stations:
        data = obs.data[time][station]
        measurements['temps'].append(data['Air temperature']['value'])
        measurements['dew_temps'].append(data['Dew-point temperature']['value'])
        measurements['hums'].append(data['Relative humidity']['value'])
        measurements['pressures'].append(data['Pressure (msl)']['value'])
        measurements['c_covers'].append(data['Cloud amount']['value'])

    result = []
    for measurement, values in measurements.items():
        if all(np.isnan(x) for x in values):
            result.append(np.nan)
        elif np.isnan(values[0]) or distances[0] > 5:
            result.append(round(np.nanmean(values), 2))
        else:
            result.append(values[0])
    return result

def pre_process_data(df):
    '''Does the preproccesing the data'''

    # Only take the actual city and not a subregion of the city
    df['city'] = df['city'].apply(lambda city: city.split(',')[0])

    # Only consider noctilucent cloud observations
    df['title'] = df['title'].str.lower()
    df = df[df['title'].str.contains('yöpil')].copy(deep=True)

    # Divide the observation into positive and negative observations
    df['title'] = df.apply(lambda row: 0 if 'neg' in row['title'] else 1, axis=1)
    df.rename(columns={'title': 'category', 'start': 'observation_time'}, inplace=True)

    # Split the coordinates into latitude and longitde and make them numeric
    df[['latitude', 'longitude']] = df['coordinates'].str.split(',', expand=True)
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])

    # Format the time 
    df['observation_time'] = df.apply(format_time, axis=1)
    
    df = df.dropna() # Drop missing data rows
    df = df.iloc[5000:5010]

    # Calculate the time from sunset
    df['time_from_sunset'] = df.apply(time_from_sunset, axis=1)

    return df

def link_weather_data(df):
    '''Links the weather obersvations into the dataframe'''

    df[['temperature', 'dew_point', 'humidity', 'pressure', 'cloud_cover']] = df.apply(get_weather_obs, axis=1, result_type='expand')
    return df

def write_data(df, file):
    '''Writes the preprocessed data into a file'''

    df['observation_time'] = df['observation_time'].apply(lambda time: time.strftime('%Y-%m-%d %H:%M'))
    # Split the date and time
    df[['observation_date', 'observation_time']] = df['observation_time'].str.split(' ', expand=True)

    df = df[['category', 'showiness', 'observation_date', 'observation_time',
            'city', 'latitude', 'longitude', 'time_from_sunset',
            'temperature', 'dew_point', 'humidity', 'pressure', 
            'cloud_cover']]
    result = df.to_json(orient="records")
    parsed = loads(result)
    with open(file, 'w') as f:
        dump(parsed, f, indent=4)

def main():
    obs = read_cloud_data('./cloud_obs_finland.json').copy(deep=True)
    obs = pre_process_data(obs)
    obs = link_weather_data(obs)
    obs = obs.dropna()
    write_data(obs, 'preprocessed_finland_test.json')

    print(obs.describe())

if __name__ == '__main__':
    main()