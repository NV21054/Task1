from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    variables = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
    return render_template('index.html', variables=variables)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[feature]) for feature in request.form]
    features = np.array(features).reshape(1, -1)

    quality_prediction = model.predict(features)
    quality_prediction = round(float(quality_prediction[0][0]), 2)
    return render_template('result.html', quality=quality_prediction)

if __name__ == '__main__':
    app.run(debug=True)
