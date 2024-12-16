from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd



df=pd.read_csv('Student_Marks.csv')
df.columns=['sinif','saat','puan']
df.head(3)


y=df[['puan']]
x=df[['sinif','saat']]
lm=LinearRegression()
model=lm.fit(x,y)

model.predict([[5,10]])