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
    features =  [float(x) for x in request.form.values()]########## change##########pipeline########
    final_features = np.reshape(np.array(features),(1,4))################# change##########pipeline#########
    prediction =np.argmax(model.predict(final_features))
    
    
    output='Error'

    if prediction==0:
        output='Iris-setosa'
    elif prediction==1:
        output='Iris-versicolor'
    else:
        output='Iris-virginica'
            

    return render_template('index.html', prediction_text='The Iris plant spcies is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
