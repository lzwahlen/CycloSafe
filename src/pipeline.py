import geopandas as gpd

#creation of road_segments.csv  
#joining accidents to roads 
#creating a high_risk label for the top 25% roads with the most accidents

#load Delft cyclist data and road segments
cyclists = gpd.read_file("../data/bron_delft_cyclists.geojson")
road_segments = gpd.read_file("../data/osm_road_segments.geojson")

#reproject both to EPSG:28992 (meter representation) 
#to buffer the roads by 20m to also get accident points very close to the road
cyclists = cyclists.to_crs("EPSG:28992")
road_segments = road_segments.to_crs("EPSG:28992")

#add a buffer to the road segments, join with roads with accidents, keep only closest segment for each accident
road_segments["geometry"] = road_segments.geometry.buffer(20)
joined = gpd.sjoin_nearest(cyclists, road_segments, how="inner")
joined = joined[~joined.index.duplicated(keep="first")] #keep only first (closest) match for each accident point

#security check:
#print(f"Accidents matched to segments: {len(joined)}")

#number of accident counts per road 
accident_counts = joined.groupby("index_right").size().rename("accident_count")
#number of roads with min. one accident
road_segments["accident_count"] = road_segments.index.map(accident_counts).fillna(0).astype(int)

#print(f"total accidents: {road_segments['accident_count'].sum()}")
#print(f"segments with min. one accident: {len(accident_counts)}")

#compute segment length in metres to normalize accident counts
road_segments["length"] = road_segments.geometry.length

#compute accident rate = accidents per metre (removes length bias)
road_segments["accident_rate"] = road_segments["accident_count"] / road_segments["length"].clip(lower=1)


#create a high-risk label, base high risk on accideent rate (instead of accident count)
#roads_accident = road_segments[road_segments["accident_count"] > 0]["accident_count"] #consider only roads with accidents
roads_accident = road_segments[road_segments["accident_count"] > 0]["accident_rate"]
threshold = roads_accident.quantile(0.75)
#road_segments["high_risk"] = (road_segments["accident_count"] > threshold).astype(int)

#more aggressive high-risk: any segment with 1 or more accidents is high risk
road_segments["high_risk"] = (road_segments["accident_count"] >= 1).astype(int)
#print(road_segments["high_risk"].sum())
#print(road_segments["high_risk"].mean())

#threshold = 1, a segment needs min. two accidents to be considered high risk
#print(f"threshold (75th percentile): {threshold}") 
#print(f"high risk segments: {road_segments['high_risk'].sum()}")
#print(f"low risk segments: {(road_segments['high_risk'] == 0).sum()}")

#reproject back for the final output for consistency
road_segments = road_segments.to_crs("EPSG:4326")
output = road_segments[["geometry", "highway", "maxspeed", "lanes", "junction", "accident_count", "accident_rate", "length", "high_risk"]].copy()


#compute centroid in projected CRS for accuracy, then extract lat/lon in 4326
centroids = output.to_crs(epsg=28992).geometry.centroid.to_crs(epsg=4326)
output["lat"] = centroids.y
output["lon"] = centroids.x

#to verify lat/lon
#print(output[["lat", "lon"]].describe())

#add segment length in metres, longer segments accumulate more exposure
output["length"] = output.to_crs(epsg=28992).geometry.length  # <-- add this line
#add accidents per metre instead of raw accident count
output["accident_rate"] = output["accident_count"] / output["length"].clip(lower=1)  # clip avoids division by zero


#save output to csv file for training
output.to_csv("../data/road_segments.csv", index=False)

#save Delft boundary for use in dashboard
import osmnx as ox
delft_boundary = ox.geocode_to_gdf("Delft, Netherlands")
delft_boundary.to_file("../data/delft_boundary.geojson", driver="GeoJSON")

