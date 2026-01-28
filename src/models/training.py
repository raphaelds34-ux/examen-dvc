import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

def train_model():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Chargement des meilleurs paramètres
    best_params = joblib.load("models/best_params.pkl")

    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/gbr_model.pkl")

if __name__ == "__main__":
    train_model()
