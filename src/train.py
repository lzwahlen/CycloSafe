from model import load_data, train_logistic_regression, train_random_forest, evaluate_model
from sklearn.model_selection import train_test_split


#load data
X,y = load_data()

#80/20 train/validation split
#use 80% of segments for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#train logistic regression
lr = train_logistic_regression(X_train, y_train)git 

#train random forest
rf = train_random_forest(X_train, y_train)

#eval models
evaluate_model(lr, X_train, y_train, "Logistic Regression")
evaluate_model(lr, X_train, y_train, "Random Forest")


