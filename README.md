# EX 04: Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages and print the present data.

2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VENKATANATHAN P R
RegisterNumber: 212223240173
*/

```python
import pandas as pd
data=pd.read_csv('Placement_data1.csv')
# print(data.head())
data1=data.copy()
data1=data.drop(['sl_no','salary'],axis=1) #removes the specified row or column
# print(data1.head())
# print(data1.isnull().sum())
# print(data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
print("\n")
print("Placement Data:")
print(data1)

x=data1.iloc[:,:-1]
print("\nThe x Values:")
print(x)

y=data1['status']
print("\nThe Status Values:")
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") # A library for large linear classification
print("\nThe Logistic Regression of Training Datasets:")
print(lr.fit(x_train,y_train))

y_pred=lr.predict(x_test)
print("\nY_prediction Array:")
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred) #Accuracy score =(tp+tn)/(tp+fn+tn+fp)
#accuracy_score(y_true,y_pred,normalize=False)
#Normalise : It contains the boolean value(ture/false).If false,return the number of column
#Othewise, it returns the fraction of correctly confidential samples
print("\nAccuracy Value:")
print(accuracy)

from sklearn.metrics import confusion_matrix
print("\nConfusion Array:")
print(confusion_matrix(y_test,y_pred)) #11+24=35 - correct predictions, 5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print("\nClassification Report:")
print(classification_report1)

print("\nPrediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### Placement Data:

![alt text](<Screenshot 2024-04-22 115445.png>)

### The x Values:

![alt text](<Screenshot 2024-04-22 115504.png>)

### The Status Values:

![alt text](<Screenshot 2024-04-22 115518.png>)

### Y_prediction Array:

![alt text](<Screenshot 2024-04-22 115534.png>)

### Accuracy Value:

![alt text](<Screenshot 2024-04-22 115543.png>)

### Confusion Array:

![alt text](<Screenshot 2024-04-22 115551.png>)

### Classification Report:

![alt text](<Screenshot 2024-04-22 115600.png>)

### Prediction of LR:

![alt text](<Screenshot 2024-04-22 115612.png>)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
