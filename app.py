from flask import Flask, jsonify, request
from keras.models import load_model
import pickle
import numpy as np

new_model = load_model('ann_model.h5')
sc = pickle.load(open('scaler.pkl','rb'))

app = Flask(__name__) # This gives each file a unique name

@app.route("/predict",methods=['POST'])
def call_to_predict():
    try:
        request_data = request.get_json()
        
        predict_req = [request_data['france'],request_data['germany'],request_data['spain'],request_data['CreditScore'],
        request_data['Gender'],request_data['Age'],request_data['Tenure'],request_data['Balance'],request_data['NumOfProducts'],
        request_data['HasCrCard'],request_data['IsActiveMember'],request_data['EstimatedSalary']]

        x_feature = np.array(predict_req)

        x_feature = np.reshape(x_feature, (1,-1))
        
        # print("this is values of something",x_feature)
        
        y_pred = new_model.predict(sc.transform(x_feature))
        
        # print("this is PREDICTED VALUE",y_pred)
        
        y_pred = y_pred[0][0]
        
        predict_json= {
            'prediction value': float(y_pred),
            'prediction':  str(y_pred < 0.5)
            }
        
        # print("this is Flatten >>>>>>>>>>>>",y_pred)
        return jsonify({"PREDICTIONS":predict_json})
    except:
        return jsonify({"An Error occured ": "Please check all input values / All inputs are mandatory"})


app.run(port=5000,debug=True)

