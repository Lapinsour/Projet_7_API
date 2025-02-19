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
    app.run(host="0.0.0.0", port=5000)
