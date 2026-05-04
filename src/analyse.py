import joblib
from model import load_data
from sklearn.model_selection import train_test_split


#load models from disk (s.t. no retraining needed)
lr = joblib.load("../models/logistic_regression.pkl")
rf = joblib.load("../models/random_forest.pkl")

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)