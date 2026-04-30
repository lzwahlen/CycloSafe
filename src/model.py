import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


def load_data():
    #load csv, fill missing values, one-hot encode, return X and y
    road_segments = pd.read_csv("data/road_segments.csv")
    #print(road_segments.shape)

    #maxspeed and lanes might have missing values
    
    #force to NaN if conversion to numerical is not possible
    road_segments["maxspeed"] = pd.to_numeric(road_segments["maxspeed"], errors="coerce") 
    road_segments["lanes"] = pd.to_numeric(road_segments["lanes"], errors="coerce")
    
    #using median to fill in missing values because mean is more dangerous with outliers
    road_segments["maxspeed"] = road_segments["maxspeed"].fillna(road_segments["maxspeed"].median())
    road_segments["lanes"] = road_segments["lanes"].fillna(road_segments["lanes"].median())

    #one-hot encoding to give every column (with text vals) numerical unique value for the model to work with
    road_segments = pd.get_dummies(road_segments, columns=["highway", "junction"])

    #define feature cols and lbls
    #exclude high_risk for features(will be predicted), also exclude geometry,lat, lon, accident_count(used for creation of high_risk)
    feature_cols = [col for col in road_segments.columns if col not in ["high_risk", "geometry", "lat", "lon", "accident_count"]]
    
    X = road_segments[feature_cols] #create feature matrix(inputs for model)
    y = road_segments["high_risk"]  #create label vector, what model is trying to predict(output)
    
    return X, y

def train_logistic_regression(X_train, y_train):
    
    #base line, simple model for binary classification: create, fit, return model
    logistic_reg_model = LogisticRegression(max_iter=500)#set max iterations to 500 to ensure convergence
    logistic_reg_model.fit(X_train,y_train)

    return logistic_reg_model

def train_random_forest(X_train, y_train):
    #create, fit, return model
    pass

def evaluate_model(model, X_test, y_test, name):
    #predict, print F1/precision/recall, return scores dict
    pass

def plot_feature_importance(model, feature_names):
    #bar chart of feature importances, save to models/
    pass

#temporary to test model functions
if __name__ == "__main__":
    temp = load_data()


