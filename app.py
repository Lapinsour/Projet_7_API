import os
import pickle
from flask import Flask, request, jsonify

# Charger le modèle
MODEL_PATH = "best_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None  # Evite que l'API crashe si le modèle est absent

app = Flask(__name__)

@app.route('/predict', methods=['POST']) #L'API attend une requête POSTE à son url/predict. 
def predict(): #Lorsqu'elle reçoit une requête POST, l'API renvoie le résultat de la fonction predict().
    try:
        data = request.get_json() #Les données reçues sont en json.
        print("🚀 Données reçues:", data)  # Log des données reçues

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Vérifie que le modèle est chargé
        if model is None:
            return jsonify({"error": "Model not found"}), 500

        # Convertir les données en DataFrame si besoin (pandas)
        import pandas as pd
        df = pd.DataFrame([data]) #Les données sont converties en df pandas parce que c'est ce qu'attend le modèle. 

        # Prédiction
        prediction = model.predict(df)
        print("🎯 Prédiction:", prediction.tolist())  # Log des prédictions

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("❌ Erreur:", str(e))  # Log des erreurs
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
