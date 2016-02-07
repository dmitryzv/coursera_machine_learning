import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


train_data = pandas.read_csv('perceptron-train.csv')
test_data = pandas.read_csv('perceptron-test.csv')

X_train = train_data.ix[:, 1:]
y_train = train_data.ix[:, 0]

X_test = test_data.ix[:, 1:]
y_test = test_data.ix[:, 0]

classifier = Perceptron(random_state=241)

classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

res = accuracy_score(y_test, y_predicted)

# Scale train and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier.fit(X_train_scaled, y_train)
y_predicted_scaled = classifier.predict(X_test_scaled)

res_for_scaled = accuracy_score(y_test, y_predicted_scaled)

print(res_for_scaled - res)
