import json
from pprint import pprint
import pandas
from pandas import DataFrame

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def test_cross_validation(clf, x, y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    bag_accuracy = cross_val_score(clf, x, y.ravel(), cv=kfold)
    print(bag_accuracy.mean())

def main():
    x_train = pandas.read_csv("feature_set_1000.csv").values
    # print(x_train)
    y_train = pandas.read_csv("target_1000.csv").values
    # print(y_train)
    x_val = pandas.read_csv("val_feature_set_100.csv").values
    y_val = pandas.read_csv("val_target_100.csv").values

    bagClf = BaggingClassifier()

    pca = PCA(n_components=25).fit(x_train)
    x_train_reduced = pca.transform(x_train)
    # test_cross_validation(bagClf, x_train_reduced, y_train)
    # test_cross_validation(bagClf, x_train, y_train)

    bagClf.fit(x_train, y_train.ravel())
    validation_accuracy = bagClf.score(x_val, y_val.ravel())
    print(validation_accuracy)



if __name__ == '__main__':
    main()