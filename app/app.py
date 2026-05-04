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

#set initial map view
view_state = pdk.ViewState(
    latitude=52.012,    #Delft centre coordinates
    longitude=4.357,
    zoom=13,            #city-level zoom (show all of Delft, zoom = 15 is street level)
    pitch=0             #flat top-down view
)

#tooltip when hover over dot
tooltip = {
    "html": """
        <b>Risk score:</b> {predicted_risk_score}<br>
        <b>Accidents:</b> {accident_count}<br>
        <b>Actual high risk:</b> {high_risk_actual}
    """,
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "13px"}
}

#render map
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"  #clean light basemap
    )
)
