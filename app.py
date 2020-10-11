import numpy as np
from flask import Flask, render_template, request
from utils import Model

model_name = 'stacked_resnet'
model = Model(model_name)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/intro')
def intro():
    return render_template('about.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get pregnancies
        gender = request.form['sex']
        if gender == 'male': preg = 0
        else:
            pregnant = request.form['pregnant']
            if pregnant == 'no':
                preg = 0
            else:
                preg = int(request.form['pregnancy'])

        # get glucose
        unit = request.form['unit']
        if unit == 'mg':
            glucose = int(request.form['sugar_level'])
        else:
            glucose = int(int(request.form['sugar_level'])*18)

        # get blood pressure
        bp = int(request.form['blood_pressure'])

        #get skin thickness
        st = int(request.form['triceps'])

        #get insulin
        insulin = int(request.form['insulin'])

        # get bmi
        bmi = float(request.form['bmi'])

        # get diabetes pedigree function
        dpf = float(request.form['pedigree'])

        #get age
        age = int(request.form['age'])
        input_data = np.array([preg, glucose, bp, st, insulin, bmi, dpf, age])
        my_prediction = model.predict(input_data)
        if my_prediction == 0:
            return render_template('result_good.html')
        else:
            return render_template('result_bad.html')
if __name__ == '__main__':
    app.run(debug=False)
