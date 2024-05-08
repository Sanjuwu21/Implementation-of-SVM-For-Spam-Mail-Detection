# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import neccessary libraries required.
2. Load the dataset using pd.read_csv.
3. Use CountVectorizer to convert text data into a matrix of token counts.
4. Create an SVM model with a linear kernel.
5. print the accuracy and classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sanjeev D
RegisterNumber: 212223040185
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as t
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv("/content/spam.csv",encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=t(x,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))
```

## Output:
![WhatsApp Image 2024-05-09 at 01 27 07_dd9d8415](https://github.com/Sanjuwu21/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146498969/93c8506e-13f5-4806-ac97-01028c4c1b4d)
![WhatsApp Image 2024-05-09 at 01 27 15_642af5e3](https://github.com/Sanjuwu21/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146498969/b66da49a-6213-4bc9-a654-f88fe2bd6575)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
