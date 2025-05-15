import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from load_data import PO_metadata_Train

occurrences = PO_metadata_Train
geometry = gpd.points_from_xy(occurrences.lon, occurrences.lat)
geo_df = gpd.GeoDataFrame(occurrences, geometry=geometry)
bounds = geo_df.total_bounds
bounds[:2] -= 1
bounds[2:] += 1
min_x, min_y, max_x, max_y = bounds
lon_list = [min_x, max_x, max_x, min_x, min_x]
lat_list = [min_y, min_y, max_y, max_y, min_y]
polygon_geom = Polygon(zip(lon_list, lat_list))
polygon = gpd.GeoDataFrame(crs='epsg:4326', geometry=[polygon_geom])
polygon.to_file("bounding_box.geojson", driver='GeoJSON')
