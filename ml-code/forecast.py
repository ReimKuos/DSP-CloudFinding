import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the map of Finland from a Shapefile
finland_map = gpd.read_file('./finland.geojson')

# Calculate the bounding box
bbox = finland_map.bounds

# Define the side length of grid squares (in degrees)
side_length = 0.1  # Adjust this value as needed

# Create an empty GeoDataFrame to store the grid squares and values
grid = gpd.GeoDataFrame(columns=['geometry', 'value'], crs=finland_map.crs)

# Create grid squares within the bounding box and assign random values
squares = []
values = []
for x in np.arange(int(bbox.minx.min()), int(bbox.maxx.max()), side_length):
    for y in np.arange(int(bbox.miny.min()), int(bbox.maxy.max()), side_length):
        square = Polygon([(x, y), (x+side_length, y), (x+side_length, y+side_length), (x, y+side_length)]
                        , crs=finland_map.crs)
        squares.append(square)
        # Generate random values (you should replace this with your data)
        value = random.uniform(0, 1)
        values.append(value)

# Combine the squares and create a GeoDataFrame
grid['geometry'] = squares
grid['value'] = values

# Filter grid squares to only include those intersecting Finland
grid = grid[grid.intersects(finland_map.unary_union)]

# Load a dataset of cities (example format)
cities = gpd.read_file('cities.shp')  # Replace 'cities.shp' with the path to your city data

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Create a heatmap using a colormap (in this case, a gradient from blue to red)
cmap = plt.get_cmap('coolwarm')  # Replace 'coolwarm' with your desired colormap
norm = plt.Normalize(vmin=0, vmax=1)  # Adjust the range based on your data

for idx, row in grid.iterrows():
    color = cmap(norm(row['value']))
    gpd.GeoSeries(row['geometry']).plot(ax=ax, color=color)

# Plot the cities on top of the heatmap
cities.plot(ax=ax, color='black', markersize=5)

# (Optional) Plot the map of Finland
finland_map.boundary.plot(ax=ax, color='red', linewidth=1)

# Set axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Grid of Squares Covering Finland as a Heatmap with Cities')

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Value')

# Display the plot
plt.show()