import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd


MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None  # Evite que l'API crashe si le modèle est absent

app = Flask(__name__)



def predict_model(data) : 
            # Convertir les données en DataFrame si besoin (pandas)            
            df = pd.DataFrame([data]) #Les données sont converties en df pandas parce que c'est ce qu'attend le modèle.     
            # Prédiction
            prediction = model.predict(df)
            print("Prédiction:", prediction.tolist())  # Log des prédictions
            if model is None:
                return jsonify({"error": "Model not found"}), 500
            else :
                return prediction.tolist()

            


@app.route('/predict', methods=['POST']) #L'API attend une requête POSTE à son url/predict. 


def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        client_id = data.get("client_id")

        if not client_id:
            return jsonify({"error": "Missing client_id"}), 400

        # 🔎 On récupère les données du client dans le df_sample
        client_data = df_sample[df_sample["client_id"] == client_id]

        if client_data.empty:
            return jsonify({"error": f"Client ID {client_id} not found"}), 404

        # 🧹 Suppression des colonnes non utilisées
        client_data = client_data.drop([
            "client_id",  # identifiant
            "TARGET", 
            "NAME_FAMILY_STATUS_Unknown", 
            "NAME_INCOME_TYPE_Maternity_leave"
        ], axis=1, errors="ignore")  # errors="ignore" si parfois absentes

        # 🧠 Prédiction avec le modèle
        prediction = model.predict(client_data)[0]
        probability = model.predict_proba(client_data)[0][1]

        # ✅ Retour du résultat
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
