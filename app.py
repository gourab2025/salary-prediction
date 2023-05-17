import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
 
    YoE = int(request.form.get('Year of Experience'))
    result = model.predict(np.array([YoE]).reshape(1, -1))
    return str(result)

    return render_template('index.html',pred='Expected Salary is{}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)