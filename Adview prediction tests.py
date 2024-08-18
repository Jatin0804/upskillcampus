import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# read the training data
data_train = pd.read_csv("train.csv")

# check the head of data
data_train.head()
data_train.tail()

# check the shape of dataset
data_train.shape

# map each category of video to a number
category = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
}
data_train["category"] = data_train["category"].map(category)

# check again the data
data_train.head()

# remove character 'F' present in data
data_train = data_train[data_train.views != 'F']
data_train = data_train[data_train.likes != 'F']
data_train = data_train[data_train.dislikes != 'F']
data_train = data_train[data_train.comment != 'F']

# convert values to integers
data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])

column_vidid = data_train["vidid"]


# sklearn library
from sklearn.preprocessing import LabelEncoder

# encode features
data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])

data_train.head()

# import time 
import datetime
import time

# convert time_in_sec for duration
def check(x):
    year = x[2:]
    hours = ''
    minutes = ''
    seconds = ''
    mm = ''
    P = ["H", "M", "S"]
    for i in year:
        if i not in P:
            mm += i
        else:
            if i == "H":
                hours = mm
                mm = ''
            elif i == "M":
                minutes = mm
                mm = ''
            else:
                seconds = mm
                mm = ''
    if hours == '':
        hours = '00'
    if minutes == '':
        minutes = '00'
    if seconds == '':
        seconds = '00'
    bp = hours + ":" + minutes + ":" + seconds
    return bp

train = pd.read_csv("train.csv")
mp = pd.read_csv("train.csv")["duration"]
time = mp.apply(check)

def func_sec(time_string):
    hours, minutes, seconds = time_string.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)

time1 = time.apply(func_sec)
data_train["duration"] = time1

data_train.head()

# visulaization
plt.hist(data_train["category"])
plt.show()

plt.plot(data_train["adview"])
plt.show()

# remove videos with adview greater than 2000000
data_train = data_train[data_train["adview"] < 2000000]

# heatmap
import seaborn as sns

f, ax = plt.subplots(figsize = (10, 8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot=True)
plt.show()

# split the data
Y_train = pd.DataFrame(data = data_train["adview"].values, columns = ['target'])
data_train = data_train.drop(["adview"], axis = 1)
data_train = data_train.drop(['vidid'], axis = 1)

data_train.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state=42)
X_train.shape

# Normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.mean()

# Evaluation metrics
from sklearn import metrics


def print_error(X_test, y_test, model_name):
    predictions = model_name.predict(X_test)
    print("Mean absolute error: ", metrics.mean_absolute_error(y_test, predictions))
    print("Mean squared error: ", metrics.mean_squared_error(y_test, predictions))
    print("Root Mean Squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# Linear Regression
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test, y_test, linear_regression)


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test, y_test, decision_tree)


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators,max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
random_forest.fit(X_train, y_train)
print_error(X_test, y_test, random_forest)


# Support vector regressor
from sklearn.svm import SVR
support_vectors = SVR()
support_vectors.fit(X_train, y_train)
print_error(X_test, y_test, support_vectors)

# Artificial neural Network
import keras
from keras.layers import Dense

ann = keras.models.Sequential([
    Dense(6, activation="relu", input_shape=X_train.shape[1:]),
    Dense(6, activation="relu"),
    Dense(1)
])

optimizer = keras.optimizers.Adam()
loss = keras.losses.mean_squared_error
ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])

print(type(X_train))
print(type(y_train))
x_train = np.array(X_train)
y1_train = np.array(y_train)
print(type(X_train))
print(type(y1_train))
print(X_train.shape)
print(y_train.shape)
print(np.isnan(y_train).sum())
print(y_train.dtype)

y_train = np.array(y_train)

history = ann.fit(X_train, y_train, epochs=100)

# saving Scikit-learn models
import joblib
joblib.dump(decision_tree, "decision_tree_yt_adview.pkl")


# saving keras ANN
ann.save("ann_yt_adview.h5")
