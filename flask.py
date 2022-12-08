import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
model = keras.models.load_model('./md.h5')

@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]########## change##########pipeline########
    final_features = [np.array(int_features)]################# change##########pipeline#########
    prediction = model.predict(final_features)

    if round(prediction[0])==0:
        output='Iris-setosa'
    elif round(prediction[0])==1:
        output='Iris-versicolor'
    else:
        output='Iris-virginica'
            

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
