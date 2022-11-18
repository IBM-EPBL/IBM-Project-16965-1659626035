import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv('heart.csv');
y=df['Heart Disease']
x=df.drop('Heart Disease',axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20)
rf=RandomForestClassifier()
rf(kernel='linear').fit(X_train,Y_train)
pickle.dump(rf,open('heartt.pkl','wb'))

