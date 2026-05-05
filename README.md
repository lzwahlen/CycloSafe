<h1 style="display: flex; align-items: center;">
  <img src="app/assets/cyclosafe_logo_bike.png" alt="Logo" width="60" style="margin-right: 15px;">
  CycloSafe
</h1>

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

To evaluate the model I calculated the achieved F1 Score, precision and recall.

- Precision: of all the segments that were predicted as high-risk, how many actually were?
- Recall: of all the segments that were actually high-risk, how many were found by the model?
- F1 Score: mean of precision and recall, summarising how well a classifier performs on an imbalanced dataset

I did not just use and show the accuracy of the model, because the dataset has 858 high risk segments out of 116625. A model that classifies every segment as low-risk would get a very high accuracy (over 99%), but would in reality be completely useless. 

| Model | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.025 | 0.013 | 0.599 |
| Random Forest | 0.027 | 0.014 | 0.552 |

The F1 Score of both models is very low. This indicates that the models are not able to reliably classify road segments as high- or low-risk. 

As seen in the feature importance, the model assigns a risk score based on different attributes of the road segments such as road features and maxspeed. These current features (for example road type) are weak proxies for the actual risk. 

For example, two roads of the same type can have different cycling traffic: one could be a quiet road behind a supermarket and another one could be on a busy lane, used by hundreds of cyclists daily. For the model they would look almost identical, which makes it hard to predict different risks for the two roads.

To fix this, I searched for another dataset that captures the amount of cyclists on the roads of Delft. However, the dataset I found (NDW FietsData) only contained data for max. 14 locations in Delft for the year 2024 (for previous years I found even less data points on the website). This is not enough, most roads would get the median fallback value. This makes the feature not useful enough for training, which is why I didn't include it in the final pipeline of the model.

I tried to compare a more complex Random Forest Model to a Logistic Regression baseline, but both models perform similarly as the limiting factor is the data, not the model complexity. 

## Future Improvements

- Number of cyclist per road data as an additional feature
  - I explored NDW FietsData, but only 14 measurement points/ locations were providing meaningful data for Delft. More cyclist counts for the streets of Delft would directly help the low F1 score by giving the model more information to learn from.
- Intersection-level modelling
  - Accidents happen more often at intersections than along roads because that is where bikes and cars interact. My current model assigns intersection accidents  to nearby road segments, creating noisy labels. Reframing the problem from road segments to intersections could produce a stronger signal on the same data.
- Extend to other municipalities 


## Tech Stack 
I built this project using a Python-based geospatial and ML stack:
- Data Processing: Pandas, OSMnx and GeoPandas for handling the road network graphs and spatial data
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
- Accident data: Sourced from the BRON database via ([data.overheid.nl] https://data.overheid.nl/)
    - Dataset version: The specific dataset used is the ongevallen_2022_2024 GeoJSON dataset.
- Road network: Sourced from ([OpenStreetMap] https://www.openstreetmap.org/)
    - Implementation: The road network data was pulled using the osmnx Python library for the regions Delft, Rijswijk, Pijnacker-Nootdorp, Midden-Delfland, Den Haag, and Westland.  


