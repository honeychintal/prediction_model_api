## prediction_model_api
this is a flask api with ann model for prediction

# Inputs must be done with 'post' request with all inputs are mandatory

## Inputs must be json in below format 
{ 
"CreditScore" :738,
"france" :0,
"germany":1,
"spain":0,
"Gender" :1,
"Age" :58, 
"Tenure" :2,
"Balance" :133745.44,
"NumOfProducts" :4,
"HasCrCard" :1,
"IsActiveMember" :0,
"EstimatedSalary" :28373.86
}

# where Gender is 1 = male, 0 = female,
(france = 1, germany = 0 , spain = 0) for Geography 'france'
(france = 0, germany = 1 , spain = 0) for Geography 'germany'
(france = 0, germany = 0 , spain = 1) for Geography 'spain'

## Prediction outputs
True = leave the bank\
False = not leaving the bank
