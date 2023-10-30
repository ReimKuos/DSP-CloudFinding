from urllib.request import urlopen
import json

start = '2000-01-01'
end = '2023-09-25'
category_id = 5
format = 'json'

base_url = 'https://www.taivaanvahti.fi/app/api/search.php?'
url = base_url + f'format={format}&start={start}&end={end}&category={category_id}&country=Fi'
response = urlopen(url)
data_json = json.loads(response.read())

with open('cloud_obs_finland.json', 'w') as f:
    json.dump(data_json, f, indent=4)
