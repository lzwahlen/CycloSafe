from model import load_data, train_logistic_regression, train_random_forest, evaluate_model, plot_feature_importance
from sklearn.model_selection import train_test_split


#load data
X,y = load_data()

#sanity check
#print(f"Positives: {y.sum()}, Total: {len(y)}, Rate: {y.mean():.4f}")
#print(X.describe())
#print(f"\nFeature columns: {X.columns.tolist()}")


#80/20 train/validation split
#use 80% of segments for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#print how many rows belong to each class:
#print(y.value_counts())
#print(y.value_counts(normalize=True))
#problem: only 198 high-risk segments out of 116625 
#sol: undersample majority class (take 5000 randomly sampled negatives and all positives)
pos_idx = y_train[y_train == 1].index
neg_idx = y_train[y_train == 0].sample(n=15000, random_state=42).index
bal_idx = pos_idx.union(neg_idx).to_series().sample(frac=1, random_state=42).index

X_train_bal = X_train.loc[bal_idx]
y_train_bal = y_train.loc[bal_idx]

#train logistic regression
lr = train_logistic_regression(X_train_bal, y_train_bal)
#lr = train_logistic_regression(X_train, y_train)

#train random forest
rf = train_random_forest(X_train_bal, y_train_bal)
#rf = train_random_forest(X_train, y_train)

#do a threshold sweep:
#print(f"\n{'Threshold':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
#for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
#    scores = evaluate_model(rf, X_test, y_test, "RF", threshold=t)
#    print(f"{t:<12} {scores['f1']:<10.3f} {scores['precision']:<12.3f} {scores['recall']:<10.3f}")

#eval models
lr_evals = evaluate_model(lr, X_test, y_test, "Logistic Regression")
rf_evals = evaluate_model(rf, X_test, y_test, "Random Forest")


#print evaluation, plot it? is fr actually better than lr? 
print(f"\n{'Model':<25} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print(f"{lr_evals['name']:<25} {lr_evals['f1']:<10.3f} {lr_evals['precision']:<10.3f} {lr_evals['recall']:<10.3f}")
print(f"{rf_evals['name']:<25} {rf_evals['f1']:<10.3f} {rf_evals['precision']:<10.3f} {rf_evals['recall']:<10.3f}")


#plot feature importance 
plot_feature_importance(rf, X_train_bal.columns.tolist())
#plot_feature_importance(rf, X_train.columns.tolist())
