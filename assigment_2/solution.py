import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
clean_data = data[['Pclass','Fare','Sex','Age', 'Survived']]
clean_data = clean_data[clean_data.Age.notnull()]
clean_data = clean_data.replace({'Sex': {'male': 1, 'female': 2}})

clf = DecisionTreeClassifier(random_state=241)
res = clf.fit(clean_data.ix[:, 0:4], clean_data[['Survived']])
clf.feature_importances_
