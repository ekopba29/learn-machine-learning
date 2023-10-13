from flask import Flask, request, jsonify
import tensorflow as tf
import os

current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, 'model')

app = Flask(__name__)
# this is a Flask application route


@app.route("/")
# this is a FLask application function
def load_balancing():
    return "<p><h3>Flask Application Load Balancing using Docker Compose and Nginx<h3></p>"


@app.route("/test")
def test():
    return "masuk mantap"


@app.route('/predict')
def predict():
    input_text = request.args.get('text')

    if not input_text:
        return jsonify({'err': 'missing parameter query text -> ?text='})

    model = tf.keras.models.load_model(model_dir)
    prediction = model.predict([input_text])
    result = {'tingkat sarkas ': f'{float(prediction[0][0]*100):.2f}%'}

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5005)
