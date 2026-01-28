import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
import os

def grid_search():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    param_grid = yaml.safe_load(open("params.yaml"))["grid_search"]

    gbr = GradientBoostingRegressor()
    grid = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_params_, "models/best_params.pkl")

if __name__ == "__main__":
    grid_search()
