import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
best_model = pickle.load(open('loanmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    output = best_model.predict('data')
    print (output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data= [float(x) for x in request.form.values()]
    final_input = data
    print(final_input)
    output = best_model.predict(final_input)[0]
    return render_template('home.html', prediction_text= "The Loan is{}".format(output))

if __name__ =="__main__":
    app.run(debug=True)
