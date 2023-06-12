import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())))
    cols = ['experience_level', 'employment_type', 'remote_ratio', 'company_size']
    for col in cols:
        data[f'EN_{col}'] = encoder.fit_transform(data[col])
    
    def EN_mean_salary(x):
        cols = ['job_title', 'employee_residence', 'company_location']
        base = 'salary_in_usd'
        for col in cols:
            x[f'EN_mean_{col}'] = data.groupby(col)[base].transform('mean')
        return x

    output = regmodel.predict(data)
    print(output)
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
