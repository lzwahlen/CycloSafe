from model import load_data, train_logistic_regression, train_random_forest, evaluate_model
from sklearn.model_selection import train_test_split


#load data
X,y = load_data()

#print how many rows belong to each class:
print(y.value_counts())
print(y.value_counts(normalize=True))

#80/20 train/validation split
#use 80% of segments for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#train logistic regression
lr = train_logistic_regression(X_train, y_train)

#train random forest
rf = train_random_forest(X_train, y_train)

#eval models
lr_evals = evaluate_model(lr, X_test, y_test, "Logistic Regression")
rf_evals = evaluate_model(rf, X_test, y_test, "Random Forest")

#print evaluation, plot it? is fr actually better than lr? 
print(f"\n{'Model':<25} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print(f"{lr_evals['name']:<25} {lr_evals['f1']:<10.3f} {lr_evals['precision']:<10.3f} {lr_evals['recall']:<10.3f}")
print(f"{rf_evals['name']:<25} {rf_evals['f1']:<10.3f} {rf_evals['precision']:<10.3f} {rf_evals['recall']:<10.3f}")
