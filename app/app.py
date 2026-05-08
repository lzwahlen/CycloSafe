import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point, LineString
import osmnx as ox
import ast

#page configuration
st.set_page_config( page_title="CycloSafe", layout="wide")#uses full browser width

#page title
st.title("CycloSafe")


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

#add metrics row at the top
col1, col2, col3, col4 = st.columns(4)
col1.metric("Road segments", f"{len(pred):,}")
col2.metric("Accident locations", f"{pred['accident_count'].gt(0).sum():,}")
col3.metric("High risk predicted", f"{pred['predicted_high_risk'].sum():,}")
col4.metric("Model", "Random Forest")

#add project description
st.markdown("""
This dashboard shows predicted cyclist accident risk across Delft road segments.

It is built using real Dutch BRON accident data and OpenStreetMap infrastructure features, using a Random Forest classifier trained on road type, speed limit and lane count to assign a risk score to every segment in the city.

The filters on the left help to explore which road segments the model flagged as high risk and where recorded accidents actually occurred.        
 """)
st.divider()  #clean horizontal line


road_paths = load_road_network()


def risk_to_colour(score):
    #maps 0-1 risk score to RGB colour
    #grey → orange → red gradient
    if score < 0.3:
        return [34, 139, 34, 160]     #green (low risk)
    elif score < 0.6:
        return [255, 140, 0, 180]     #orange (medium risk)
    else:
        return [220, 20, 20, 200]     #red (high risk)

#apply color to every row
pred["colour"] = pred["predicted_risk_score"].apply(risk_to_colour)


#add sidebar title and slider
#st.sidebar.image("app/assets/cyclosafe_logo_bike.png", width=80)
#st.sidebar.markdown("### CycloSafe")
col1, col2 = st.sidebar.columns([1, 3])
col1.image("app/assets/cyclosafe_logo_bike.png", width=80)
col2.markdown("### CycloSafe")


st.sidebar.markdown("Cyclist risk prediction, Delft (2022 - 2024)")
st.sidebar.divider()

st.sidebar.header("Filters")

#risk score slider, filters which segments are shown on the map
min_risk = st.sidebar.slider(
    label="Minimum risk score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,        #default shows all segments
    step=0.05
)

#add accident count slider
min_accidents = st.sidebar.slider(
    label="Minimum accident count",
    min_value=0,
    max_value=int(pred["accident_count"].max()),
    value=0,          #default shows all segments
    step=1
)

#filter the dataframe based on slider values
pred_filtered = pred[
    (pred["predicted_risk_score"] >= min_risk) &
    (pred["accident_count"] >= min_accidents)
]

#add filter by road type
#check if highway is in predictions.csv
#print(pred.columns.tolist())

#clean highway column first:
#each road segment has multiple highway tags stored as a list
#take only first/primary road type
#because if multiple names, finds no match 
#def clean_highway_display(val):
#    try:
#        parsed = ast.literal_eval(val)
#        if isinstance(parsed, list):
#            return parsed[0]
#        return val
#    except:
#        return str(val).strip("[]'\"").split("'")[0].strip()

#pred["highway"] = pred["highway"].astype(str).apply(clean_highway_display)

#nice display names for the cleaned values
#road_type_labels = {
#    "service": "service road",
#    "residential": "residential street",
#    "cycleway": "cycleway",
#    "tertiary": "tertiary road",
#    "secondary": "secondary road",
#    "primary": "primary road",
#    "living_street": "living street",
#    "pedestrian": "pedestrian zone",
#    "unclassified": "unclassified road",
#    "path": "path",
#    "track": "track",
#    "busway": "busway",
#    "bridleway": "bridleway",
#    "trunk": "trunk road",
#    "primary_link": "primary link",
#    "secondary_link": "secondary link",
#    "tertiary_link": "tertiary link",
#    "trunk_link": "trunk link"
#}

#get unique road types from original highway column in road_segments.csv
road_types = sorted(pred["highway"].dropna().unique().tolist())

#map types to display names
#display_names = [road_type_labels.get(rt, rt) for rt in road_types]

selected_road_types = st.sidebar.multiselect(
    label="Filter by road type",
    options=road_types,
    default=[]  # empty default means all road types shown
)
#selected_road_display = st.sidebar.multiselect(
#    label="Filter by road type",
#    options=display_names,
#    default=[]  #empty default means all road types shown
#)


#convert back to raw values for filtering
#display_to_raw = {v: k for k, v in road_type_labels.items()}
#selected_road_types = [display_to_raw.get(d, d) for d in selected_road_display]

#apply road type filter, if nothing selected show all
if selected_road_types:
    pred_filtered = pred_filtered[pred_filtered["highway"].isin(selected_road_types)]

#info line for both sliders
st.sidebar.markdown(f"Showing **{len(pred_filtered):,}** of **{len(pred):,}** segments")



tab1, tab2 = st.tabs(["Risk Map", "Model Insights"])

with tab1:
    #pydeck layer (scatterplotLayer places one dot per row at [lon, lat])
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pred_filtered,  #filtered instead of full pred
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
        data=pred_filtered[pred_filtered["accident_count"] > 0],
        get_position=["lon", "lat"],
        get_fill_color=[30, 144, 255, 180],  #blue
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

    else:
        layers = [layer_roads, layer_accidents]
        tooltip = tooltip_accidents


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

    #color guide based on selection
    if view_mode == "Predicted risk score":
        st.markdown("""
        **Risk level colour guide**
        - Red: high predicted risk (score > 0.6)
        - Orange: medium predicted risk (score 0.3–0.6)
        - Green: low predicted risk (score < 0.3)
        """)
    else:
        st.markdown("""
        **Accident locations colour guide**
        - Blue: road segment with at least one recorded accident
        """)


with tab2:

    st.subheader("What did the model learn?") #show feature importance and shap plot for model insight

    #feature importance
    st.markdown("#### Feature importance")
    st.caption("How much each feature influenced the model's decisions across all trees.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.image("plots/feature_importance.png", caption="Feature Importance Plot", width= 600 )

    with col2:
        st.markdown("""
        Each bar represents the how much the specific feature was used to split the data across all decision points in the Random Forest. 
        All these values add up to 1.
                    
        **Key findings:**
        - `highway_service` (27%) is the strongest predictor 
        - `maxspeed` (16%) confirms the model learned physically plausible signals
        - `highway_cycleway` (7.5%) suggests cycle infrastructure is meaningful 

        Note: importance only tells you *how much* a feature mattered, not whether it pushed the risk up or down.
        """)
    

    #shap summary
    st.markdown("#### SHAP Plot")
    st.caption("How each feature pushed the individual predictions towards a higher or lower risk than the average risk score.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.image("plots/shap_summary.png", caption="SHAP Plot", width= 600 )

    with col2:
        st.markdown("""
        Each point represents a road segment. 
        Its position on the x-axis shows how much the feature pushed the risk score up (positive) or down (negative).
        """)
        st.markdown("""
        The colour of the point shows the feature value (**red = high value**, **blue = low value**)""")
        st.markdown("""     
        **For example:** A blue point for `maxspeed` on the negative side means a low speed limit pushed that segment towards a lower risk.
        """)
        st.markdown("""""")
        st.markdown("""
        **Key Findings:**
        - Cycleways increase the predicted risk.
        - Less lanes and service roads push the predicted risk towards lower values
        - A higher maximum speed can increase or decrease the predicted risk.
                
        """)

    #add limitations to the app to show what model can and cannot do
    st.markdown("#### Limitations")

    #limitation 1: missing cyclist volume
    #st.markdown("""**Missing cyclist volume**""")
    st.markdown("""
    >**Missing cyclist volume:** OSM road attributes cannot distinguish a quiet dead-end from a busy route. 
    Even though they are not equally dangerous, the model cannot see a difference between those roads and therefore also cannot predict different risk scores.
    """)
    #st.markdown(""" """)

    #limitation 2: BRON accident underregistration
    #st.markdown("""**BRON accident underregistration**""")
    st.markdown("""
    >**BRON accident underregistration**: Solo cyclist falls and smaller incidents are rarely reported, so the accident counts per segment are a lower bound on the true risk.
    """)
    #st.markdown(""" """)

    #limitation 3: Spatial limitation
    #st.markdown("""**Spatial limitation**""")
    st.markdown("""
    >**Spatial limitation**: OSMnx pulled a slightly larger area than Delft city limits. 
    This leads to segments near the boundary maybe having incomplete accident records.
    """)


#add shap chart? 

#st.markdown("""
#**Risk level colour guide**
#- Red — high predicted risk (score > 0.6)
#- Orange — medium predicted risk (score 0.3–0.6)
#- Grey — low predicted risk (score < 0.3)
#- Blue — road segment with at least one recorded accident
#""")

