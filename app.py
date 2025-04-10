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
df_sample = pd.read_csv("df_sample.csv")
df_sample = df_sample.drop(['TARGET', 'NAME_FAMILY_STATUS_Unknown', 'NAME_INCOME_TYPE_Maternity_leave'], axis=1, errors='ignore')



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
        if client_id is None:
            return jsonify({"error": "client_id is missing"}), 400

        # 🔎 On récupère les données du client
        client_data = df_sample[df_sample['SK_ID_CURR'] == client_id]

        if client_data.empty:
            return jsonify({"error": "Client not found"}), 404

        # 🚫 Supprimer 'client_id' avant la prédiction
        client_data = client_data.drop(['SK_ID_CURR'], axis=1, errors='ignore')

        # 🤖 Faire la prédiction
        prediction = model.predict_proba(client_data)[0]

        return jsonify({
            "prediction": prediction.tolist()  # ou juste prediction[1] si tu veux la proba de la classe 1
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
