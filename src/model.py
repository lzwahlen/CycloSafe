import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import ast
from sklearn.calibration import calibration_curve


def clean_highway(val):
    #each road segment has multiple highway tags stored as a list
    #take only first/primary road type

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed[0]
        return val
    except:
        return str(val).strip("[]'\"").split("'")[0].strip()


def load_data():
    #load csv, fill missing values, one-hot encode, return X and y
    road_segments = pd.read_csv("../data/road_segments.csv")
    road_segments = road_segments.drop(columns=["Unnamed: 0"], errors="ignore")

    #safety check:
    #print(road_segments.shape)
    #print(f"high risk segments: {road_segments['high_risk'].sum()}")


    #maxspeed and lanes might have missing values, fill in by taking the median fallback value
    
    #strip list wrapper (for exaaple: go from ['50'] to 50) before calling to numeric 
    #(otherwise it wouldn't recognise numbers and turn everything to NaN)
    road_segments["maxspeed"] = road_segments["maxspeed"].str.extract(r'(\d+)')[0]
    road_segments["lanes"] = road_segments["lanes"].str.extract(r'(\d+)')[0]

    #force to NaN if conversion to numerical is not possible
    road_segments["maxspeed"] = pd.to_numeric(road_segments["maxspeed"], errors="coerce") 
    road_segments["lanes"] = pd.to_numeric(road_segments["lanes"], errors="coerce")

    #using median to fill in missing values because mean is more dangerous with outliers
    road_segments["maxspeed"] = road_segments["maxspeed"].fillna(road_segments["maxspeed"].median())
    road_segments["lanes"] = road_segments["lanes"].fillna(road_segments["lanes"].median())

    #one-hot encoding to give every column (with text vals) numerical unique value for the model to work with
    #clean highway column before encoding (removes multi-value combinations)
    road_segments["highway"] = road_segments["highway"].astype(str).apply(clean_highway)
    road_segments = pd.get_dummies(road_segments, columns=["highway", "junction"])


    #define the feature columns and labels

    #exclude high_risk for features(will be predicted), also exclude geometry,lat, lon, accident_count(used for creation of high_risk)
    feature_cols = [col for col in road_segments.columns if col not in ["high_risk", "geometry", "lat", "lon", "accident_count", "accident_rate", "length"]]

    X = road_segments[feature_cols] #create feature matrix(inputs for model)
    y = road_segments["high_risk"]  #create label vector, what model is trying to predict(output)
    
    return X, y


#logistic regression: baseline model
def train_logistic_regression(X_train, y_train):
    #added class_weight=balanced to penalize missing minority class more heavily (without it would predict every road segement as low risk (always high_risk = 0, never 1))
    logistic_reg_model = LogisticRegression(max_iter=2000, class_weight= 'balanced')#set max iterations to 500 to ensure convergence
    logistic_reg_model.fit(X_train,y_train)

    #save the model to the disk
    joblib.dump(logistic_reg_model, "../models/logistic_regression.pkl")

    return logistic_reg_model


#random forest model: handles messy real-world data well (no normalization/scaling necessary)
#it also handles class imbalance better than simpler models
def train_random_forest(X_train, y_train):
    #build 200 trees, take majority vote, make result reproducible with random_state = 42 (every run same model)
    #added class_weight = balanced to penalize missing minority class more heavily (without it would predict every road segement as low risk (always high_risk = 0, never 1))
    rand_forest_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight= 'balanced') 
    rand_forest_model.fit(X_train, y_train)

    joblib.dump(rand_forest_model, "../models/random_forest.pkl")

    return rand_forest_model


def evaluate_model(model, X_test, y_test, name, threshold = 0.5):
    #compute F1 score, precision and recall

    #use predict_proba() to manually change threshold
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    #use f1 score to balance recall and precision(find most dangerous segments, while not having too many false alarms)
    f1 = f1_score(y_test, y_pred) # = 2/((1/precision) + (1/recall))
    
    #precision = how many of predicted high risk row_seg were actually high risk
    precision = precision_score(y_test, y_pred) # = 1 - false positive rate

    #recall = of all the segments that were actually high-risk, how many were found by the model?
    recall = recall_score(y_test, y_pred) # = true positive rate

    return {"name": name, "f1": f1, "precision": precision, "recall": recall}


def evaluate_probabilities(model, X_test,y_test,name):
    #additional plot to further evaluate the probabilities (calibration plot)

    #predicted class probabilities of model, (for a map of gradients)
    prob = model.predict_proba(X_test)
    probs_class1 = prob[:, 1]  #take only class 1 (high risk) probs

    #plot the calibration of the model

    #calculate the fraction of positives and the mean predicted value for the plot
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probs_class1, n_bins=10
    )

    #plot the computed values in the calibration plot
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives,
             marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual positives")
    plt.title(f"Calibration Plot: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../plots/calibration_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()


def eval_tree_convergence(rf, X_train, y_train, X_test, y_test ):
    #plot the tree convergence (Random Forest equivalent of a loss curve)

    #use different amount of trees
    tree_counts = [10, 25, 50, 100, 150, 200, 300]
    f1_scores = []

    #sweep over all tree counts and compute the F1 scores
    for n in tree_counts:
        rf = RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        f1_scores.append(f1_score(y_test, preds, zero_division=0)) #add the F1 scores to the list for the plot

    #plot the gathered F1 scores
    plt.figure(figsize=(8, 4))
    plt.plot(tree_counts, f1_scores, marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("F1 score")
    plt.title("F1 score vs number of trees: Random Forest")
    plt.tight_layout()
    plt.savefig("../plots/rf_convergence.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names):
    #bar chart of the feature importances to evaluate which features influenced the Random Forest model's decisions the most
    
    #compute feature importance score from every feature from the trained Random Forest model
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:15] #use top 15 only for a cleaner chart
    
    #plot the top 15 feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(15), importances[indices])
    plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title("Random Forest — Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("../plots/feature_importance.png", dpi=150)
    plt.close() 



