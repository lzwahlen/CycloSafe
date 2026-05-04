import streamlit as st
import pandas as pd
import pydeck as pdk


#page configuration
st.set_page_config( page_title="CycloSafe", layout="wide")#uses full browser width

#page title
st.title("CycloSafe")
st.markdown("Predicted cyclist accident risk on Delft road segments.")

#load data data
#load file once, cache it (without it, file reloads every time user interacts with anything)
@st.cache_data

def load_predictions():
    pred = pd.read_csv("data/predictions.csv")
    return pred

pred = load_predictions()

#pydeck layer (scatterplotLayer places one dot per row at [lon, lat])
layer = pdk.Layer(
    "ScatterplotLayer",
    data=pred,
    get_position=["lon", "lat"],
    get_fill_color=[220, 20, 20, 160],  #fixed red for all dots
    get_radius=15,
    pickable=True
)

#set initial map view
view_state = pdk.ViewState(
    latitude=52.012,    #Delft centre coordinates
    longitude=4.357,
    zoom=13,            #city-level zoom (show all of Delft, zoom = 15 is street level)
    pitch=0             #flat top-down view
)

#render map
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9"  #clean light basemap
    )
)
