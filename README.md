# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load and Prepare Data**
   Import required libraries, load the dataset (`food_items_binary.csv`), and select the features (`Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein`) and target variable (`class`).

2. **Split and Scale the Data**
   Divide the dataset into training and testing sets using `train_test_split`, then apply `StandardScaler` to normalize the feature values.

3. **Train SVM with GridSearchCV**
   Initialize the **SVM classifier (SVC)** and use **GridSearchCV** to find the best parameters (`C`, `kernel`, `gamma`) with cross-validation.

4. **Evaluate the Model**
   Predict the test data using the best model and calculate **accuracy, classification report, and confusion matrix**, then visualize it using a **heatmap**.


## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import seaborn as sns

data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat', 'Sugars','Dietary Fiber','Protein' ]
target='class'

X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_

print("Name: PRIYADHARSHINI")
print("Register Number:212225220076")
print("Best Parameters:",grid_search.best_params_)

y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: PRIYADHARSHINI")
print("Register Number:212225220076")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

![WhatsApp Image 2026-03-28 at 4 07 38 PM](https://github.com/user-attachments/assets/d2f48a56-251e-4450-a7b3-4fb80b8d9c19)


![WhatsApp Image 2026-03-28 at 4 07 58 PM](https://github.com/user-attachments/assets/4fee2253-3521-4aa0-9b7e-79cf5218c09a)

![WhatsApp Image 2026-03-28 at 4 08 19 PM](https://github.com/user-attachments/assets/15ad7eed-eae5-4b79-be78-3776a19f289d)


![WhatsApp Image 2026-03-28 at 4 08 40 PM](https://github.com/user-attachments/assets/77b25ce0-28c9-42ce-b34e-c8c6403791e5)


![WhatsApp Image 2026-03-28 at 4 09 13 PM](https://github.com/user-attachments/assets/993443a8-5ce2-45a1-872e-753fb0950f15)


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
