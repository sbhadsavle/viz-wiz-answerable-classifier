import json
from pprint import pprint
import pandas
from pandas import DataFrame

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def main():
	x_train = pandas.read_csv("feature_set_300.csv").values
	# print(x_train)
	y_train = pandas.read_csv("target_300.csv").values
	# print(y_train)

	
	bagClf = BaggingClassifier()
	# bagClf.fit(x_train, y_train.ravel())

	kfold = KFold(n_splits=3, shuffle=True, random_state=0)

	bag_accuracy = cross_val_score(bagClf, x_train, y_train.ravel(), cv=kfold)
	print(bag_accuracy)


if __name__ == '__main__':
    main()