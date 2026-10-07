import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, plot_importance, plot_tree

# Run classification on breast cancer detection
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

#Split trainign and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, train_size=0.7, random_state=42, stratify=y
)

#Train XGBoost Classifier 
model_clf = XGBClassifier(
    #determines the number of trees
    n_estimators=100,
    #maximum number of levels or splits in each decision tree
    max_depth=5,
    #step size at each iteration
    learning_rate=0.1,
    #specify learning task and loss function
    objective='reg:squarederror',
    #seed state
    random_state=42
)

model_clf.fit(X_train, y_train)

#Predict and evaluate
y_pred = model_clf.predict(X_test)
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("Confustion Matrix:")
print(cm)

### ------------------------------------------------------------- ###

# Hyperparameter tuning with GridSearchCV (Classification)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best cross validation score: {grid_search.best_score_}")