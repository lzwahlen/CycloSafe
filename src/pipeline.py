import geopandas as gpd


#load Delft cyclist data and road segments
cyclists = gpd.read_file("data/bron_delft_cyclists.geojson")
road_segments = gpd.read_file("data/osm_road_segments.geojson")

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

#create a high-risk label 
roads_accident = road_segments[road_segments["accident_count"] > 0]["accident_count"] #consider only roads with accidents
threshold = roads_accident.quantile(0.75)
road_segments["high_risk"] = (road_segments["accident_count"] > threshold).astype(int)

#threshold = 1, a segment needs min. two accidents to be considered high risk
#print(f"threshold (75th percentile): {threshold}") 
#print(f"high risk segments: {road_segments['high_risk'].sum()}")
#print(f"low risk segments: {(road_segments['high_risk'] == 0).sum()}")

#reproject back for the final output for consistency
road_segments = road_segments.to_crs("EPSG:4326")
output = road_segments[["geometry", "highway", "maxspeed", "lanes", "junction", "accident_count", "high_risk"]].copy()
output["lat"] = output.geometry.centroid.y
output["lon"] = output.geometry.centroid.x

#save output to csv file for training
output.to_csv("data/road_segments.csv", index=True)
