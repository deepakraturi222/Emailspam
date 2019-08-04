import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
data=pd.read_csv("spam.csv")
print(data.describe())
x,y=data["EmailText"],data["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print(y_test)
cv=CountVectorizer() #extract features
feature=cv.fit_transform(x_train)
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-5],
                     'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(feature,y_train)

print(model.best_params_)

#model=svm.SVC()
#model.fit(feature,y_train)
f_test=cv.transform(x_test)
print(model.score(f_test,y_test))
