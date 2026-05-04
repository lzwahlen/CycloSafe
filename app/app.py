import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point, LineString
import osmnx as ox

#page configuration
st.set_page_config( page_title="CycloSafe", layout="wide")#uses full browser width

#page title
st.title("CycloSafe")
st.markdown("Predicted cyclist accident risk on Delft road segments.")

#load data 
#load file once, cache it (without it, file reloads every time user interacts with anything)
@st.cache_data
def load_predictions():
    pred = pd.read_csv("data/predictions.csv")

    #filter to Delft 
    #filter with bounding box
    #pred = pred[
    #    (pred["lat"] >= 51.95) & (pred["lat"] <= 52.05) &
    #    (pred["lon"] >= 4.30) & (pred["lon"] <= 4.40)
    #]

    #filter with delft region border
    #load Delft boundary from disk — no network request needed
    delft_boundary = gpd.read_file("data/delft_boundary.geojson")
    delft_polygon = delft_boundary.geometry.iloc[0]

    #check which segment centroids fall inside the boundary
    geometry = [Point(lon, lat) for lon, lat in zip(pred["lon"], pred["lat"])]
    gdf = gpd.GeoDataFrame(pred, geometry=geometry, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(delft_polygon)]

    return gdf


#for map of roads underneath use path data of osm geojson file
@st.cache_data
def load_road_network():
    roads = gpd.read_file("data/osm_road_segments.geojson")
    roads = roads.to_crs("EPSG:4326")
    
    #PathLayer needs coordinates as a list of [lon, lat] pairs
    #extract them from the geometry
    paths = []
    for geom in roads.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            paths.append({"path": [[lon, lat] for lon, lat in geom.coords]})
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                paths.append({"path": [[lon, lat] for lon, lat in line.coords]})
    
    return paths

pred = load_predictions()

#round predicted risk score
pred["predicted_risk_score"] = pred["predicted_risk_score"].round(3)

road_paths = load_road_network()


def risk_to_colour(score):
    #maps 0-1 risk score to RGB colour
    #grey → orange → red gradient
    if score < 0.3:
        return [150, 150, 150, 160]   #grey (low risk)
    elif score < 0.6:
        return [255, 140, 0, 180]     #orange (medium risk)
    else:
        return [220, 20, 20, 200]     #red (high risk)

#apply color to every row
pred["colour"] = pred["predicted_risk_score"].apply(risk_to_colour)


#pydeck layer (scatterplotLayer places one dot per row at [lon, lat])
layer = pdk.Layer(
    "ScatterplotLayer",
    data=pred,
    get_position=["lon", "lat"],
    get_fill_color="colour",        #column name for colour
    get_radius=15,
    pickable=True
)

#path layer for the roads
layer_roads = pdk.Layer(
    "PathLayer",
    data=road_paths,
    get_path="path",
    get_color=[180, 180, 180, 80],  #light grey, semi-transparent
    get_width=10,                     #line width in metres
    pickable=False
)

#layer to highlight only segments where accidents actually happened
layer_accidents = pdk.Layer(
    "ScatterplotLayer",
    data=pred[pred["accident_count"] > 0],
    get_position=["lon", "lat"],
    get_fill_color=[255, 255, 0, 230],  #yellow
    get_radius=25,                       #larger so they sit visibly on top
    pickable=True
)

#set initial map view
view_state = pdk.ViewState(
    latitude=52.012,    #Delft centre coordinates
    longitude=4.357,
    zoom=13,            #city-level zoom (show all of Delft, zoom = 15 is street level)
    pitch=0             #flat top-down view
)

#tooltip when hover over dot for the predictions
tooltip_pred = {
    "html": """
        <b>Risk score:</b> {predicted_risk_score}<br>
        <b>Accidents:</b> {accident_count}<br>
        <b>Actual high risk:</b> {high_risk_actual}
    """,
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "13px"}
}


#tooltip for the actual accidents
tooltip_accidents = {
    "html": """
        <b>Accidents recorded:</b> {accident_count}<br>
        <b>Predicted risk score:</b> {predicted_risk_score}<br>
    """,
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "13px"}
}

#add sidebar toggle
view_mode = st.radio(
    "Map display mode",
    options=["Predicted risk score", "Actual accident locations"],
    horizontal=True  #displays options side by side instead of stacked
)


#layers based on selection
if view_mode == "Predicted risk score":
    layers = [layer_roads, layer]
    tooltip = tooltip_pred
    st.markdown("""
    **Risk level colour guide**
    - Red: high predicted risk (score > 0.6)
    - Orange: medium predicted risk (score 0.3–0.6)
    - Grey: low predicted risk (score < 0.3)
    """)
else:
    layers = [layer_roads, layer_accidents]
    tooltip = tooltip_accidents
    st.markdown("""
    **Accident locations**
    - Yellow: road segment with at least one recorded accident
    """)


#render map
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        #map_style="mapbox://styles/mapbox/light-v9"  #clean light basemap
    )
)


#add sidebar filter for risk score?
#add shap chart? 

#st.markdown("""
#**Risk level colour guide**
#- Red — high predicted risk (score > 0.6)
#- Orange — medium predicted risk (score 0.3–0.6)
#- Grey — low predicted risk (score < 0.3)
#- Yellow — road segment with at least one recorded accident
#""")
