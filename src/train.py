from src.model import load_data
from sklearn.model_selection import train_test_split


#load data
X,y = load_data()

#80/20 train/validation split
#use 80% of segments for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


