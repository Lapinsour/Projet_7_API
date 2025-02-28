import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/echo', methods=['POST'])
def echo():
    try:
        data = request.get_json()  # Récupérer les données JSON envoyées

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Vérifier si toutes les colonnes attendues sont présentes
        if len(data) != 121:
            return jsonify({"error": f"Expected 121 features, but got {len(data)}"}), 400

        return jsonify({"received_data": data})  # Renvoie exactement ce qui a été reçu

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
