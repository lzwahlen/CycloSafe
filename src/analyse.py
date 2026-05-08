import joblib
from model import load_data, evaluate_probabilities
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


#load models from disk (s.t. no retraining needed)
lr = joblib.load("../models/logistic_regression.pkl")
rf = joblib.load("../models/random_forest.pkl")

X, y = load_data()
#recreate data split: 80/10/10 train/validation/test 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


#use SHAP explainer for Random Forest
#for each segment and each feature their SHAP value = contribution of that feature to pushing the prediction away from the average prediction
#(negative SHAP value means feature pushed prediction toward low risk)
explainer = shap.TreeExplainer(rf)

#run on a 200-sample subset of test set (full test set is slow)
X_test_sample = X_test.sample(n=200, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

#shape of shap vals
#print(type(shap_values))
#print(np.array(shap_values).shape)

#shap_values is a list of two arrays [class_0, class_1] (class 1 = high risk class)
shap_values_class1 = shap_values[:, :, 1]

#plot SHAP values
plt.figure(figsize=(12, 8)) #increase figure size 
shap.summary_plot(shap_values_class1, X_test_sample, max_display=15, show=False)  #show only top 15 features 
plt.title("SHAP Summary: Feature Impact on High-Risk Prediction", fontsize=13, pad=15)
plt.xlabel("SHAP value (impact on predicted risk)", fontsize=11)
plt.tight_layout()
plt.savefig("../plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

road_segments_full = pd.read_csv("../data/road_segments.csv")

#(for more honest f1 score by testing on unseen geographic areas of Delft instead of random segments)
#data from the same neighbourhood is similar, in 80/20 train/test split data of same neighborhood end up in 
#train and test data which makes the model already have seen train data very similar to the test data
#sol: spatial crossvalidation (-> explicit geographical separation)

#spatial cross validation:

#load raw csv data (including lat and lon) and reset indices
road_segments_full = pd.read_csv("../data/road_segments.csv")
#reset indices to ensure allignment
road_segments_full = road_segments_full.reset_index(drop=True)
X_reset = X.reset_index(drop=True)
y_reset = y.reset_index(drop=True)

#define center point and quadrant function for the geographical data split
LAT_MID = 52.012
LON_MID = 4.357

def get_quadrant(lat, lon):
    if lat >= LAT_MID and lon >= LON_MID:
        return 0  #north east
    elif lat >= LAT_MID and lon < LON_MID:
        return 1  #north west
    elif lat < LAT_MID and lon >= LON_MID:
        return 2  #south east
    else:
        return 3  #south west

#assign quadrant to every road segment
road_segments_full["quadrant"] = road_segments_full.apply(
    lambda row: get_quadrant(row["lat"], row["lon"]), axis=1
)

#safety check, ensure good quadrant distribution
#print(road_segments_full["quadrant"].value_counts().sort_index())
#print(road_segments_full.groupby("quadrant")["high_risk"].sum())

spatial_f1_scores = []

#rotation loop, four times per quadrant
#always one quadrant in test set and other three in training set
for q in range(4):
    test_mask = (road_segments_full["quadrant"] == q).values
    train_mask = ~test_mask

    #apply geographic mask to/split features and labels
    X_spatial_train = X_reset[train_mask]
    y_spatial_train = y_reset[train_mask]
    X_spatial_test = X_reset[test_mask]
    y_spatial_test = y_reset[test_mask]

    #skip empty quadrants
    if y_spatial_test.sum() == 0:
        print(f"skip quadrant {q}, no high-risk segments")
        continue

    #train each time a new model
    rf_spatial = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_spatial.fit(X_spatial_train, y_spatial_train)

    #evaluate on test quadrant
    y_pred = rf_spatial.predict(X_spatial_test)
    f1 = f1_score(y_spatial_test, y_pred, zero_division=0)
    spatial_f1_scores.append(f1)

y_pred_random = rf.predict(X_test)
random_split_f1 = f1_score(y_test, y_pred_random, zero_division=0)

#output results of spatial cross validation and compare to random split
mean_spatial_f1 = np.mean(spatial_f1_scores)
print(f"Mean spatial CV F1:  {mean_spatial_f1:.3f}")
print(f"Random split F1:     {random_split_f1:.3f}")
print(f"Performance drop:    {random_split_f1 - mean_spatial_f1:.3f}")

#calibration plot to check if probability scores are actually meaningful
evaluate_probabilities(rf, X_test,y_test,"Random Forest")
evaluate_probabilities(lr, X_test,y_test,"Logistic Regression")


#save predictions
#get risk scores for all segments, not just test set (what the dashboard map will use)
all_probs = rf.predict_proba(X)[:, 1]  # continuous risk score 0-1
all_preds = (all_probs >= 0.5).astype(int)

#(add highway for selection in dashboard)
predictions = pd.DataFrame({ "lat": road_segments_full["lat"].values, "lon": road_segments_full["lon"].values, "predicted_risk_score": all_probs,
    "predicted_high_risk": all_preds, "highway": road_segments_full["highway"].values, "high_risk_actual": y.values, "accident_count": road_segments_full["accident_count"].values})

#check if highway is in predictions.csv
#print(predictions.columns.tolist())
#print(sorted(predictions["highway"].dropna().unique().tolist()))


predictions.to_csv("../data/predictions.csv", index=False)

#sanity check
#print(predictions.shape)
#print(predictions.head())
#print(f"Risk score range: {all_probs.min():.3f} to {all_probs.max():.3f}")
#print(f"Predicted high risk: {all_preds.sum()}")
