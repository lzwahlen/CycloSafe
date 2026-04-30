import geopandas as gpd


#Load Delft cyclist data and road segments
cyclists = gpd.read_file("data/bron_delft_cyclists.geojson")
road_segments = gpd.read_file("data/osm_road_segments.geojson")

#Reproject both to EPSG:28992 (meter representation) 
#to buffer the roads by 20m to also get accident points very close to the road
cyclists = cyclists.to_crs("EPSG:28992")
road_segments = road_segments.to_crs("EPSG:28992")


