import sklearn.datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import numpy

sklearn.preprocessing.scale

boston_housing = sklearn.datasets.load_boston()

boston_housing.data = scale(boston_housing.data)

kfold = KFold(len(boston_housing.target), n_folds=5, shuffle=True, random_state=42)

res_p = None
max_accuracy = -100

for p in numpy.linspace(1, 10, num=200):
    classifier = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    accuracy = cross_val_score(classifier,
                               boston_housing.data,
                               boston_housing.target,
                               cv=kfold,
                               scoring='mean_squared_error').mean()
    print(accuracy)

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        res_p = p

print(res_p, max_accuracy)
