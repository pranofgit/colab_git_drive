import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('./md.h5')

@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.values()########## change##########pipeline########
    final_features = [np.array(features)]################# change##########pipeline#########
    #prediction = model.predict(final_features)
    prediction=0###############change this#########
    
    output='Error'

    if round(prediction)==0:
        output='Iris-setosa'
    elif round(prediction)==1:
        output='Iris-versicolor'
    else:
        output='Iris-virginica'
            

    return render_template('index.html', prediction_text='The Iris plant spcies is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
