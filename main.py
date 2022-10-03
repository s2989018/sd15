import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("jaaa")

df = pd.read_csv('bodyy.csv', sep=';')
#print(df.describe().to_string())

df=df.replace({'M':0, 'F':1})
df=df.replace({'A':1,'B':2,'C':3,'D':4})

#is for when u wanna see all columns
#print(df.to_string())

#Is to see the row with value 0
#print(df.loc[df['systolic'] < 60].to_string())

# to drop all rows with a zero
#df= df.dropna()

print(df['weight_kg'].unique)

# To replace zero value ,in column systolic, with the mean
systolic_nep = df['systolic']
systolic_nep.replace(to_replace = 0, value = systolic_nep.mean(), inplace=True)

df['gender'].value_counts()

X=df.drop(columns=['class'])
y=df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

prediction = regressor.predict(X_test)
mse = mean_squared_error(y_test, prediction)
print("mse is:" ,mse)
rmse = np.sqrt(mse)
print("rmse is:",rmse)
# corr = df.corr()
# sb.heatmap(corr, cmap="Blues", annot=True)
# plt.show()
# dataplot=sb.heatmap(df.corr())
# plt.show()


#print(df.isnull().sum())
#print(df.dtypes)
# df.plot(kind = 'scatter', x = 'age', y= 'weight_kg')
# df['age'].plot(kind = 'hist')
# plt.show()
#median1 = df['age'].median()


