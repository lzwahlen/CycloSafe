<img align="left" width="50" src="app/assets/cyclosafe_logo_bike.png" alt="Project Logo">

# CycloSafe

CycloSafe predicts accident hotspots for cyclists on road segments in Delft (Netherlands) using real Dutch open data.

## What I Built

I developed an end-to-end ML pipeline to analyse the risk of bike accidents in Delft. Joining the BRON accidents dataset (2022-2024) with OpenStreetMap road infrastructure data (retrieved via OSMnx), the model predicts the risk of a bike accident happening on a specific road segment. I compared a Logistic Regression baseline with a Random Forest classifier to evaluate performance and analysed how different road infrastructure features correlate with the number of accidents happening. The final results are shown in an interactive Streamlit dashboard with a Pydeck risk map to help visualise high-danger zones and accident locations.

### Dashboard Preview

The dashboard is fully interactive, allowing users to filter road segments by type and explore high-risk areas based on a minimum risk score and accident count.

![CycloSafe Dashboard Interface](app/assets/cyclosafe_dashboard.png)
*Figure 1: The dynamic CycloSafe dashboard is designed to help users explore and filter predicted collision risks on the streets of Delft.*

![CycloSafe Map of Actual Accidents](app/assets/cyclosafe_dashboard_accidents.png)
*Figure 2: The CycloSafe Actual Accidents Map shows the collision locations across the road network. Hovering over a specific data point reveals more information on the risk score and the amount of accidents that occured.*

## Key findings

*to do: add general findings and thoughts, add screenshot of model insight of dashboard*


### Temp (!) notes on plots
### feature importance plot

![Feature Importance - Top 15 feature importances](plots/feature_importance.png)

Findings:
- highway_cycleway at 0.075: very counterintuitive, I would have expected cycling infrastructure to reduce risk, not predict it, but maybe this is because they have more cyclists and therefore more accidents, this comes from the data limitation introduced by BRON as BRON records accidents but not cyclist volume, so high-use infrastructure looks risky even when it is well designed
- highway_service at 0.267: service roads (access roads, parking aisles, driveways, and back-of-building roads) are most predictive features, surprising that on service roads most cyclist accidents happen, maybe because service roads are very common in osm network or because they are tight spaces with no cycling infrastructure
- maxspeed at 0.160: speed limit second strongest signal, makes sense since higher speed creates more severe accidents
- lanes at 0.090: more lanes generally means busier, wider roads, shap plot showed that in some cases wider lanes push toward lower risk which could be the case because wider roads in Delft might have better cycling infrastructure or clearer lane separation

In total, the plot shows that the model learned something real from the data, even if it is not strong enough to classify reliably. 

It explains why F1 is 0.027. In a dataset where features strongly separate classes the top feature should be at 0.5 or higher. Even the best feature only explains about a quarter of the model's decisions which means no single feature is a reliable predictor on its own.


### shap plot

![SHAP Summary: Feature Impact on High-Risk Prediction](plots/shap_summary.png)

Even though model prediction is not that good and F1 score is very low, I still used the SHAP plot. SHAP does not evaluate whether the model is accurate, but it explains what the model learned from the data it had. 

The model learned that cycleways are associated with higher predicted risk based on patterns in the training data.

These findings should be treated as preliminary until the model's predictive performance improves.

- each dot in plot represents one road semgent
- horizontal position = how strongly that feature pushed predicted risk score up (right) or down (left)
- dot color = segments actual feature value (e.g. for cycleways: red = 1 -> segment is cycleway, blue = 0 -> segment is not a cycleway)

Findings:
- highway_cycleway has many dots to the right, cycleways are associated with higher predicted risk, makes sense since there is more cycling in general
- junction_roundabout has one red dot at -0.45, being roundabout = very low predicted risk, makes sense since dutch roundabout separates cyclists from cars (safe)
- highway_tertiary has strongest positve val (0.15), signals strongest risk push, maybe because moderate speed with cars and bikes mixed -> dangerous
- highway_service has widest overall spread -> affects more segments than any other feature, but per-segment influence is weak
- violet dots in maxspeed: segments have medium speed limit, neither highest nor lowest in dataset -> present but maybe unreliable 

## Results

 *to do: add and discuss results*

## Tech Stack 
I built this project using a Python-based geospatial and ML stack:
- Data Processing: OSMnx and GeoPandas for handling the road network graphs and spatial data
- Machine Learning: scikit-learn for the Random Forest classifier and Logistic Regression baseline, SHAP for interpreting feature importance
- Visualisation: Streamlit for the interactive dashboard and Pydeck for high-performance spatial mapping

## Setup:
Prerequisites:
- Ensure you have Conda or Mamba installed
- Ensure you have an active internet connection (to fetch OSM data)

Clone the repository: 
```bash
git clone https://github.com/lzwahlen/CycloSafe.git
cd CycloSafe
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cyclosafe
```

Run the pipeline and train the model:
```bash
cd src
python pipeline.py
python train.py
python analyse.py
```

Run the dashboard:
```bash
cd ..
streamlit run app/app.py      
```


## Notes

- Logo: CycloSafe bike logo drawn by me
- Accident data: Sourced from the BRON database via data.overheid.nl (www.data.overheid.nl)
    - Dataset version: The specific dataset used is the ongevallen_2022_2024 GeoJSON dataset.
- Road network: Sourced from OpenStreetMap (www.openstreetmap.org)
    - Implementation: The road network data was pulled using the osmnx Python library for the regions Delft, Rijswijk, Pijnacker-Nootdorp, Midden-Delfland, Den Haag, and Westland.  


