import pytest
import pandas as pd
import requests
from app import predict_model

# URL brute du fichier CSV sur GitHub 
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Lapinsour/Projet_7_API/refs/heads/main/df_sample.csv"

def load_random_data():    
    df = pd.read_csv(GITHUB_CSV_URL)
    df = df.drop('TARGET', axis=1)
    # Sélectionner une ligne aléatoire et convertir en dict
    return df.sample(n=1).iloc[0].to_dict()

def test_predict_model():
    data = load_random_data()  # Récupérer une ligne aléatoire du CSV
    result = predict_model(data)
    assert 0 <= result <= 1
