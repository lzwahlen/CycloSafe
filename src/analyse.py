import joblib
from model import load_data
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import matplotlib.pyplot as plt

#load models from disk (s.t. no retraining needed)
lr = joblib.load("../models/logistic_regression.pkl")
rf = joblib.load("../models/random_forest.pkl")

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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

