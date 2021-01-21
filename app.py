from flask import Flask
from flask import request
import pandas as pd
import pickle
import json
import os
from preprocess_data import preprocess

app = Flask(__name__)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return 'Welcome to No-Show Inference Server'


@app.route('/postjson', methods=['GET' , 'POST'])
def postJsonHandler():
    my_dict = {}
    my_list = ['Gender', 'Age', 'Neighbourhood', 'Scholarship',
               'Hipertension',
               'Diabetes',
               'Alcoholism',
               'Handcap',
               'SMS_received',
               'ScheduledDay_DOW',
               'AppointmentDay_DOW',
               'ScheduledDay_Month',
               'AppointmentDay_Month',
               'ScheduledDay_Year',
               'AppointmentDay_Year',
               'Num_App_Missed']

    res = json.loads(request.json)
    for i in my_list:
        values = res[i].values()
        my_dict.update({i: values})

    df = pd.DataFrame(my_dict)
    X = df.to_numpy()
    y_pred = model.predict(X)

    return str(y_pred)


@app.route('/predict', methods=["POST"])
def predict():
    print('type of the request ', type(request))
    print('request is json', request.is_json)
    print(request)
    print(request.get_json())
    print(type(request))

    # X = pd.DataFrame(json.loads(request.get_json()))
    X = pd.read_json(request.get_json())
    X = preprocess(X)
    y_pred = model.predict_proba(X)[:, 1]
    return json.dumps(list(y_pred))


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
