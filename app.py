import pickle
from flask import Flask, request, app, jsonify, url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
           
def home():
        return render_template('home.html')

@app.route('/predict_api',method=['POST'])

def predict_api():
        data = request.json['data']
        print(data)
        print(np.array(list(data.value())))
        cols = ['experience_level', 'employment_type', 'job_title','salary_currency','employee_residence','company_location','company_size']
        data[cols]=data[cols].apply(encoder().fit_transform)
        output = regmodel.predict(data)
        print(output)
        return jsonify(output)

if __name__ =='__main__':
        app.run(debug=True)