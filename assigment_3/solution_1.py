import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

def get_res(wine_class, wine_treits, kfold):
    max_accuracy = -1
    res_n_neighbours = None

    for n_neighbors in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        accuracy = cross_val_score(classifier, wine_treits, wine_class, cv=kfold).mean()

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            res_n_neighbours = n_neighbors

    return res_n_neighbours, max_accuracy


data = pandas.read_csv('wine.data')

wine_class=data.ix[:, 0]
wine_treits=data.ix[:, 1:]

kfold = KFold(len(wine_class), n_folds=5, shuffle=True, random_state=42)

print(get_res(wine_class, wine_treits, kfold))

print(get_res(wine_class, scale(wine_treits), kfold))
