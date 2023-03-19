from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

loaded_model = load_model('advance_model.h5') # Chargement du modèle

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    sequence = request.json['sequence']
    while len(sequence[0]) < 35:
        sequence[0].insert(0, 0)
    prediction = loaded_model.predict(sequence)
    stringprediction = str(prediction[0][0])
    return jsonify({'prediction': stringprediction})

if __name__ == '__main__':
    app.run()