import os
 import pickle
 from flask import Flask, request, jsonify
 import pandas as pd
 
 # Charger le modèle
 
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
 
 
 def predict(): #Lorsqu'elle reçoit une requête POST, l'API renvoie le résultat de la fonction predict().
     try:
         data = request.get_json() #Les données reçues sont en json.       
 
         if not data:
             return jsonify({"error": "No data received"}), 400      
 
         return jsonify({"prediction": predict_model(data)})
 
     except Exception as e:
 
         return jsonify({"error": str(e)}), 500
 
 if __name__ == '__main__':
     port = int(os.environ.get("PORT", 5000))
     app.run(host="0.0.0.0", port=port)
