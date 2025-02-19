import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    if "value" in data and isinstance(data["value"], (int, float)):
        return jsonify({"result": data["value"] > 0})

    return jsonify({"error": "Invalid input"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # On récupère le port depuis l'environnement
    app.run(host="0.0.0.0", port=port)  # 0.0.0.0 permet d'accepter les connexions externes
