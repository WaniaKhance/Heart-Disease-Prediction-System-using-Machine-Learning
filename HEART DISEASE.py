from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn import tree
#DATAPREPROCESSING
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", 
         "ca", "thal", "target"]
dataset = pd.read_csv('dataset.csv', names=names, header=None)
dataset.replace("?", np.nan, inplace=True)
#print(dataset.isnull().sum())  Summing how many NaN values are in each column
dataset.dropna(axis=0, inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)
a = (dataset.target == 1).sum()
b = (dataset.target == 0).sum()
plot2 = pd.DataFrame({'Target':['No diesease', 'Disease'], 'Count':[a, b]})
ax = plot2.plot.bar(x='Target', y='Count', rot=0)
ax.set_xlabel("Target", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("People with heart disease")

X=dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
Y=dataset['target']

# RANDOM FOREST REGRESSION MODEL
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 60)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_train,y_train )
predictions = rf.predict(x_test)
x = 0
for i in range (1,100):
    x = x + r2_score(y_test, predictions)
r2 = x/100
print("R2 score for Random Forest regression",r2*100)

#11111 LINEAR REGRESSOR
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.3,random_state = 60)
lr = LinearRegression()
lr.fit(X_train2,Y_train2)
predicted2 = lr.predict(X_test2)
y = 0 
for i in range (1,100):
    y = y + r2_score(Y_test2, predicted2)
r1 = y/100
print("R2 score for Linear regression",r1*100)

#2222 RANDOM FORESTN CLASSIFIER
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3,random_state = 60)
model= RandomForestClassifier()
model.fit(X_train1,Y_train1)
predicted1= model.predict(X_test1)
a = 0
b = 0 
for i in range (1,100):
    a = a + accuracy_score(Y_test1,predicted1)
    b = b + model.score(X_train1, Y_train1)
test_accuracy1 = a/100
score3 = b/100   
print("Training Accuracy for random forest",score3*100)
print("Testing Accuracy for random forest",test_accuracy1*100)
print ("Training Error for random forest", (1 - score3)*100)
print ("Testing Error for random forest", (1-test_accuracy1)*100)



X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.3,random_state = 60)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train3, Y_train3)
predictions3 = clf.predict(X_test3)
a1 = 0
b1 = 0 
for i in range (1,100):
    a1 = a1 + accuracy_score(Y_test3,predictions3)
    b1 = b1 + model.score(X_train3, Y_train3)
test_accuracy3 = a1/100
score1 = b1/100  
print("\nTraining Accuracy for Descision Tree:",score1*100)
print("Testing Accuracy for Descision Tree:",test_accuracy3*100)
print ("Training Error for Descision Tree", (1 - score1)*100)
print ("Testing Error for Descision Tree", (1-test_accuracy3)*100)


#3333 K MEAN CLASSIFIER
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state = 60)
k_means = KMeans(n_clusters=1,random_state=0)
k_means.fit(X_train,Y_train)
predicted= k_means.predict(X_test)
test_accuracy=accuracy_score(Y_test,predicted)
print("\nTesting Accuracy for Kmean",test_accuracy*100)




#x = dataset.age
#y = dataset.target
#data_dict = {'target':pd.Series(y),'age':pd.Series(x)}
#plot1 = pd.DataFrame(data_dict)
#ax1 = plot1.plot.bar(x='target', y='age', rot=0)
#ax1.set_xlabel("Target", fontsize=12)
#ax1.set_ylabel("Age", fontsize=12)
#ax1.set_title("Disease Probability")




#plot1 = pd.DataFrame({'Age':['20', '40', '60', '80', '100'], 'Count':[a1,b1,c1,d1,e1]})
#ax1 = plot1.plot.bar(x='Age', y='Count', rot=0)
#ax1.set_xlabel("Age", fontsize=12)
#ax1.set_ylabel("Count", fontsize=12)
#ax1.set_title(" Disease probability")



