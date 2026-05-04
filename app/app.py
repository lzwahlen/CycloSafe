import streamlit as st
import pandas as pd

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