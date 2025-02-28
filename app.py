import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Charger le modèle
MODEL_PATH = "best_model.pkl"  # Chemin de ton modèle
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)  # Charger le modèle LightGBM

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Récupérer les données JSON envoyées

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Transformer en DataFrame
        df = pd.DataFrame([data])  

        # Remplacer NaN et valeurs infinies
        df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

        # Vérifier que le nombre de colonnes est correct
        if df.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Expected {model.n_features_in_} features, but got {df.shape[1]}"}), 400

        # Faire la prédiction
        prediction = model.predict(df)[0]  # Prendre la première valeur

        return jsonify({"prediction": float(prediction)})  # Convertir en JSON-safe format

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
