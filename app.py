from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import joblib 
from pathlib import Path


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':

        try:
            #  reading the inputs given by the user
            gender = request.form['gender']
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            ever_married = request.form['ever_married']
            work_type = request.form['work_type']
            Residence_type = request.form['residence_type']
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = request.form['smoking_status']

            model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
            encoders = joblib.load(Path('artifacts/data_transformation/encoders.pkl'))
            scaler= joblib.load(Path('artifacts/data_transformation/MinMaxScalar.pkl'))

            gender_encoded = encoders['gender'].transform([gender])[0]
            ever_married_encoded = encoders['ever_married'].transform([ever_married])[0]
            work_type_encoded = encoders['work_type'].transform([work_type])[0]
            residence_type_encoded = encoders['Residence_type'].transform([Residence_type])[0]
            smoking_status_encoded = encoders['smoking_status'].transform([smoking_status])[0]

            # Scale numerical values
            scaled_data = scaler.transform([[avg_glucose_level, bmi]])
            avg_glucose_level_scaled = scaled_data[0][0]
            bmi_scaled = scaled_data[0][1]              
         
            data = [gender_encoded, hypertension, heart_disease, ever_married_encoded, work_type_encoded, residence_type_encoded, avg_glucose_level_scaled, bmi_scaled, smoking_status_encoded]
            
            data = np.array(data).reshape(1, -1)
            
            predict = model.predict_proba(data)[0][1]            

            return render_template('index.html', prediction = str(round((predict * 100) , 2)))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0' , port=  5001)