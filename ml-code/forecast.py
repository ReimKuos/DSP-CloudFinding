import datetime as dt
import pytz as tz
from fmiopendata.wfs import download_stored_query
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke
import joblib
from astral.sun import sun
from astral import LocationInfo
import noise


def import_model():
    '''Imports the regression model and corresponding scaler'''

    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def synthetic_fcast(start, lon, lat, l, temp_range):
    '''Creates a synthetic weather forecast for temperature and cloud cover'''

    helsinki_tz = tz.timezone('Europe/Helsinki')
    utc_tz = tz.timezone('UTC')
    valid_times = [start+ dt.timedelta(hours=i) for i in range(6)]

    vals = {}
    for time in valid_times:
        # The synthetic forecast is based on a persistant 2D noise function
        temp = noise.snoise2(lat / l, lon / l, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=42)
        temp = (temp_range[1]-temp_range[0])*(temp+1)/2+temp_range[0]
        cover = noise.snoise2(lat / l, lon / l, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=42)
        cover = 8*(cover+1)/2
        time = time.replace(tzinfo=utc_tz).astimezone(helsinki_tz)
        vals[time] = {'temperature': temp, 'cloud_cover': cover}
        
    return vals

def fcast(start, end, lon, lat, l):
    '''Parses weather forecast data from the FMI database'''

    date_format = '%Y-%m-%d %H:%M:%S'
    helsinki_tz = tz.timezone('Europe/Helsinki')
    utc_tz = tz.timezone('UTC')

    start = start.strftime(date_format)
    end = end.strftime(date_format)
    model_data = download_stored_query("fmi::forecast::harmonie::surface::grid",
                                       args=["starttime=" + start,
                                             "endtime=" + end,
                                             f"bbox={lon},{lat},{lon+2*l},{lat+l}"])
            
    latest_run = max(model_data.data.keys())
    data = model_data.data[latest_run]
    data.parse(delete=True)
    valid_times = data.data.keys()

    vals = {}
    for time in valid_times:
        hum = np.mean(data.data[time][2]['2 metre relative humidity']["data"])
        cover = np.mean(data.data[time][10]['Total Cloud Cover']["data"])
        cover = (cover/100)*8 # Scales the values to appropriate range
        time = time.replace(tzinfo=utc_tz).astimezone(helsinki_tz)
        vals[time] = {'humidity': hum, 'cloud_cover': cover}
        
    return vals

def time_from_sunset(lon, lat, time):
    '''Calculates the time in hours from or to the nearest sunset for the given
    location and datetime. If the sunset has already happened, time is positive,
    otherwise it is negative. If the sun doesn't set at the given location, time from
    or to the midnight is calculated instead.'''

    loc = LocationInfo(latitude=lat, longitude=lon)
    prev_day = time - dt.timedelta(days=1)
    next_day = time + dt.timedelta(days=1)

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

def get_grids(country_map, start, model, scaler, synthetic=True):
    '''Generates the grid of squares covering the given map and attaches model
    predictions to each of the squares.'''
    
    bbox = country_map.total_bounds
    l = 0.25

    if not synthetic:
        today = dt.datetime.today()
        start = dt.datetime.combine(today, dt.time(start.hour, 0))
        end = start + dt.timedelta(hours=6)

    squares = []
    props = {}
    for x in np.arange(bbox[0], bbox[2], 2*l):
        for y in np.arange(bbox[1], bbox[3], l):
            square = Polygon([(x, y), (x+2*l, y), (x+2*l, y+l), (x, y+l)])
            squares.append(square)
            if synthetic:
                weather = synthetic_fcast(start, x, y, l, (10,30))
            else:
                weather = fcast(start, end, x, y, l)

            for time in weather:
                if time not in props:
                    props[time] = []
                    
                tfss = time_from_sunset(x, y, time)
                if tfss < 0:
                    props[time].append(0)
                    continue
                month_sin = np.sin(2 * np.pi * time.month / 12)
                month_cos = np.cos(2 * np.pi * time.month / 12)
                X = pd.DataFrame({
                    'longitude': [x],
                    'latitude': [y],
                    'time_from_sunset': [tfss],
                    'temperature': [weather[time]['temperature']],
                    'cloud_cover': [weather[time]['cloud_cover']],
                    'month_sin': [month_sin],
                    'month_cos': [month_cos]
                })
                X = scaler.transform(X)
                prop = model.predict_proba(X)[:, 1][0]
                props[time].append(prop)

    grids = {}
    for time in props:
        grid = gpd.GeoDataFrame({'geometry': squares, 'propability': props[time]})
        grid = grid[grid.intersects(country_map.unary_union)]
        grids[time] = grid
    return grids

def get_map():
    '''Returns the map of Finland and a curated list of cities as GeoDataFrames'''

    finland = gpd.read_file('./finland.geojson')
    cities = pd.read_json('./fi_cities.json')
    filtered_c = ['Helsinki', 'Turku', 'Tampere', 'Vaasa', 'Kuopio', 'Joensuu', 'Oulu',
                'Rovaniemi', 'Tornio', 'Kajaani', 'Jyväskylä', 'Riihimäki', 'Kotka',
                'Lappeenranta', 'Mikkeli', 'Kokkola', 'Seinäjoki', 'Pori', 'Kiuruvesi',
                'Kemijärvi', 'Inari', 'Enontekiö', 'Sodankylä', 'Kuusamo']
    cities = cities[cities['city'].isin(filtered_c)]
    cities = gpd.GeoDataFrame(cities.city, geometry=gpd.points_from_xy(cities.lng, cities.lat))

    finland.crs = "EPSG:4326"
    cities.crs = "EPSG:4326"

    return finland, cities

def plot(finland, cities, grids, time):
    '''Plots the map of propabilities'''

    date_format = '%Y-%m-%d %H:%M:%S'
    formatted_time = time.strftime(date_format)
    grid = grids[time]
    print(grid['propability'])
    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = plt.get_cmap('RdPu')
    norm = plt.Normalize(vmin=0, vmax=1)

    for idx, row in grid.iterrows():
        color = cmap(norm(row['propability']))
        gpd.GeoSeries(row['geometry']).plot(ax=ax, color=color)

    finland.boundary.plot(ax=ax, color='black', linewidth=1)

    border_effect = withStroke(linewidth=1, foreground="black")
    cities.plot(ax=ax, marker='o', color='red', markersize=5)
    for x, y, city_name in zip(cities.geometry.x, cities.geometry.y, cities['city']):
        plt.text(x, y, city_name, fontsize=11, ha='center', c='gold', path_effects=[border_effect])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Forecast for {formatted_time}')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Noctilucent cloud index')
    plt.savefig(f'{formatted_time}.png')

    output_json_file = f'{formatted_time}.json'
    grid.to_file(output_json_file, driver='GeoJSON')

def main():
    model, scaler = import_model()
    finland, cities = get_map()

    start = dt.datetime(2023, 8, 15, 18, 0)
    grids = get_grids(finland, start, model, scaler)

    for time in grids:
        plot(finland, cities, grids, time)

if __name__ == '__main__':
    main()