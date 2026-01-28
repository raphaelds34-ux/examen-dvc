import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
import os

# Chargement des paramètres
params = yaml.safe_load(open("params.yaml"))["split"]

def split_data():
    # Lecture des données (URL ou fichier local si téléchargé avec dvc get-url)
    df = pd.read_csv("data/raw/raw.csv")

    X = df.select_dtypes(include=['float64', 'int64']).drop("silica_concentrate", axis=1)
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"], 
        random_state=params["random_state"]
    )

    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    split_data()
