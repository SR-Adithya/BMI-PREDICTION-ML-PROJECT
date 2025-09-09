import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sbn
import numpy as np

# retreiving dataset from the file

file = pd.read_csv("build/bmi_edited.csv")
print(file.describe())

# find bmi formula

#valueBMI = file["Weight"]/file["Height (m)"]**2

# assign x and y columns

x = file[["Height (m)","Weight"]]
y = file["BMI"]

# data separation for training and testing

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=42)

# Start and train the model

model = LinearRegression()
model.fit(train_x,train_y)

# Predit the model with given data

pred_y = model.predict(test_x)

# Analysing Mean squared error and R2 score

print("\nMean Squared error: ", mean_squared_error(test_y,pred_y))
accuracy = r2_score(test_y, pred_y)
print(f"\nAccuracy (R2 Score): {accuracy:.2f}")

# finding intercept and coefficients of the model with the given dataset

print(f"\nIntercept: {model.intercept_}")

print("\nModel Coefficients:\n")
for features, coef in zip(x.columns, model.coef_):
    print(f"{features}: {coef:.2f}\n")

# adding color palette

palette_color = sbn.color_palette("crest", as_cmap=True)
colors = palette_color(np.linspace(0,1,2))

# plotting the graph for visualization
plt.figure(figsize=(12, 10))
plt.scatter(test_y, pred_y, color = colors[0], alpha=0.6, label='Datapoint')
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], color = colors[1], linewidth=2, label='Regression line')
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Actual vs Predicted BMI")
plt.legend()
plt.show()

# BMI Classification
print("The BMI predication analysis is based on the Indian region. The Ranges of BMI are classified by expers from ICMR and WHO\n")
print("The BMI Ranges and it's description as follows:")
bmi_table = {"Underweight": "<18.0", "Normal": "18.0 - 22.9", 
             "Overweight (At Risk)":"23.0 - 24.9", "Obese class I":"25.0 - 29.9",
             "Obese class II (High Risk)":">=30.0"}

print("\nBMI Table (Indian Standard)")
for types, bmi_range in bmi_table.items():
    print(f"{types:25}:{bmi_range}")

# prediction for new input values

BMI_test = [[153,64.8]]
height_cm, weight = BMI_test[0]

# convert height cm to m

height_m = height_cm/100

#input value validation
if height_m <=0 or weight <=0:
    print("Please enter proper Height and weight values")
else:
    BMI = model.predict([[height_m,weight]])[0]
    print("The BMI predication analysis is based on the Indian region. The Ranges of BMI are classified by expers from ICMR and WHO\n")
    print("The BMI Ranges and it's description as follows:")
    print("Weight unit is: Kg (Kilogram)\n")
    print("Height unit is: m^2 (meter squared)\n")
    print("The BMI unit is: kg per m^2\n")
    print(f"The given weight is: {weight} Kg")
    print(f"The given height is: {height_cm} cm ({height_m:.2f} m)")
    print(f"The Predicted value is: {BMI:.1f} kg/m^2\n")

# BMI range obtained and it's description
    def bmi_desc(bmi):
        if bmi<18.0:
            return "Underweight"
        elif 18.0 <= bmi <= 22.9:
            return "Normal"
        elif 23.0 <= bmi <= 24.9:
            return "Overweight (At Risk)"
        elif 25.0 <= bmi <= 29.9:
            return "Obese class I"
        elif bmi >= 30.0:
            return "Obese class II (High Risk)"
        else:
            return "Invalid BMI"
    
    category = bmi_desc(BMI)
    print(f"The predicted BMI class is: {category}\n")
    
    
    
# Alternate method Decision Tree Regressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sbn
import numpy as np


# retreiving dataset from the file

file = pd.read_csv("build/bmi_edited.csv")
print(file.describe()) # for statistical overview

# assign x and y columns

x = file[["Height (m)","Weight"]]
y = file["BMI"]

# data separation for training and testing

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.35, random_state=42)

# Now train Decision Tree Regressor model

DTR_model = DecisionTreeRegressor(random_state=42)
DTR_model.fit(train_x, train_y)

# Predict the output with the above trained model
DTR_pred_y = DTR_model.predict(test_x)

# Model evaluation
print("\n **Decision Tree Regressor Results**")
print("\nThe Mean Squared Error of Model is: ",mean_squared_error(test_y, DTR_pred_y))
DTR_accuracy = r2_score(test_y,DTR_pred_y)
print(f"\nThe Accuracy of the Model (R2 score) is: {DTR_accuracy:.2f}")

# Decision Tree Regressor don't have linear coefficients, hence skipped intercept/coefficients

# adding color palette

palette_color = sbn.color_palette("viridis", as_cmap=True)
colors = palette_color(np.linspace(0,1,2))

# Plot visualization of the DTR model
plt.figure(figsize=(12, 10))
plt.scatter(test_y, DTR_pred_y, color = colors[0], alpha=0.6, label='Datapoint')
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], color = colors[1], linewidth=2, label='Regression line')
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Decision Tree Prediction - Actual vs Predicted BMI")
plt.legend()
plt.show()

# BMI Classification
print("The BMI predication analysis is based on the Indian region. The Ranges of BMI are classified by expers from ICMR and WHO\n")
print("The BMI Ranges and it's description as follows:")
bmi_table = {"Underweight": "<18.0", "Normal": "18.0 - 22.9", 
             "Overweight (At Risk)":"23.0 - 24.9", "Obese class I":"25.0 - 29.9",
             "Obese class II (High Risk)":">=30.0"}

print("\nBMI Table (Indian Standard)")
for types, bmi_range in bmi_table.items():
    print(f"{types:25}:{bmi_range}")

# prediction for new input values

BMI_test = [[153,64.8]]
height_cm, weight = BMI_test[0]

# convert height cm to m

height_m = height_cm/100

#input value validation
if height_m <=0 or weight <=0:
    print("Please enter proper Height and weight values")
else:
    BMI = model.predict([[height_m,weight]])[0]
    print("The BMI predication analysis is based on the Indian region. The Ranges of BMI are classified by expers from ICMR and WHO\n")
    print("The BMI Ranges and it's description as follows:")
    print("Weight unit is: Kg (Kilogram)\n")
    print("Height unit is: m^2 (meter squared)\n")
    print("The BMI unit is: kg per m^2\n")
    print(f"The given weight is: {weight} Kg")
    print(f"The given height is: {height_cm} cm ({height_m:.2f} m)")
    print(f"The Predicted value is: {BMI:.1f} kg/m^2\n")

# BMI range obtained and it's description
    def bmi_desc(bmi):
        if bmi<18.0:
            return "Underweight"
        elif 18.0 <= bmi <= 22.9:
            return "Normal"
        elif 23.0 <= bmi <= 24.9:
            return "Overweight (At Risk)"
        elif 25.0 <= bmi <= 29.9:
            return "Obese class I"
        elif bmi >= 30.0:
            return "Obese class II (High Risk)"
        else:
            return "Invalid BMI"
    
    category = bmi_desc(BMI)
    print(f"The predicted BMI class is: {category}\n")
    