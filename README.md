# BMI-PREDICTION-ML-PROJECT

# Comparative Study

Body Mass Index Prediction using Multiple Linear Regression vs Decision Tree Regression

## About Dataset
Name: BMI Data, based on Indian region. Columns used: 'Height (m), 'Weight', 'BMI'
NO. of Rows used to learn the model: 10000 rows

## Model Used and Why it is used:
1) Multiple Linear Regression (MLR): Easy to interpret with statistical measures, consideres linear relation equation (y = mx + c)
2) Decision Tree Regressor (DTR): Handles complex pattern associated with data, used as an alternative for MLR

## Training method of data:
Training method: Train/test/split and test size used is 0.35 along with shuffle data using random_state(42)

## Metrics used for Evaluation:
1) MLR: {"Mean Squared Error" : 1.7696643079335126}
        {"Accuracy" : 0.98}
        {"Coefficients": {"Height (m)": -31.47, "Weight": 0.40}}

3) DTR: {"Mean Squared Error" : 0.014949935846471123}
        {"Accuracy" : 1.00}

Interpretation: As per the data learned by the model, the above statistics would resemble similar, yet when the above model trained with another huge records, then MLR would be more suitable than DTR model due to the higher R2 score which would result in overfitting of the model.

# Accuracy differences:

Accuracy of Models: 
                    MLR: 0.98, DTR: 1.00
Interpretation: The difference between MLR and DTR accuracy is found to be 0.002 which tends the DTR model to overfit with the given learning data.

## Strenghts and Weakness of the model
1) MLR:
        Strengths: Easy to deploy, can be analysed with coefficient and intercept, easily relates the data with a formula
        Weakness: Consideres linearity between the columns of trained data, could show constant error variance when data is huge

2) DTR:
        Strengths: easily handles complex relations between the data columns, less data is enough to learn
        Weakness: Often overfits the data, Lack of Robustness.

## Possible improvements / real-world applications:

Improvement: The feature columns could be increased such as work hours, sleep duration, family health history.

real-world applications: First level analysis in Hospitals, BMI check before joining a gym, Public awareness creation to follow a proper diet that would reduce percentage category of diabetic people 

## Conclusion of the project:

From the above analysis, it is recommended to add more features which are mentioned in the improvement section that would more accurately analyse the BMI index using MLR rather than DTR model especially to avoid overfitting of the model to the learning data.
