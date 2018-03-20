import json
from pprint import pprint
import pandas
from pandas import DataFrame

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def test_cross_validation(clf, x, y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    bag_accuracy = cross_val_score(clf, x, y.ravel(), cv=kfold)
    return bag_accuracy.mean()

def output_my_results(results):
    with open("my_results_30.csv", "w") as f:
        f.write("results\n")
        for res in results:
            f.write(str(res) + "\n")

def main():
    print("Starting main")
    x_train = pandas.read_csv("feature_set_1000.csv").values
    # print(x_train)
    y_train = pandas.read_csv("target_1000.csv").values
    # print(y_train)
    x_val = pandas.read_csv("val_feature_set_100.csv").values
    y_val = pandas.read_csv("val_target_100.csv").values

    x_test = pandas.read_csv("test_feature_set_30.csv").values

    bagClf = BaggingClassifier()
    adaboostClf = AdaBoostClassifier(n_estimators=200, learning_rate=0.01)
    rfClf = RandomForestClassifier(n_estimators=200)
    vClf = VotingClassifier(estimators=[('bag', bagClf), ('ada', adaboostClf), ('rf', rfClf)], voting='hard')


    # pca = PCA(n_components=25).fit(x_train)
    # x_train_reduced = pca.transform(x_train)
    # test_cross_validation(bagClf, x_train_reduced, y_train)
    # test_cross_validation(bagClf, x_train, y_train)

    bagClf.fit(x_train, y_train.ravel())
    bagClf_validation_accuracy = bagClf.score(x_val, y_val.ravel())
    print("Bagging Classifier accuracy: " + str(bagClf_validation_accuracy))

    adaboostClf.fit(x_train, y_train.ravel())
    adaboostClf_validation_accuracy = adaboostClf.score(x_val, y_val.ravel())
    print("AdaBoostClassifier accuracy: " + str(adaboostClf_validation_accuracy))

    rfClf.fit(x_train, y_train.ravel())
    rfClf_validation_accuracy = rfClf.score(x_val, y_val.ravel())
    print("RandomForestClassifier accuracy: " + str(rfClf_validation_accuracy))

    vClf.fit(x_train, y_train.ravel())
    vClf_validation_accuracy = vClf.score(x_val, y_val.ravel())
    print("VotingClassifier accuracy: " + str(vClf_validation_accuracy))

    y_pred_bag = bagClf.predict(x_test)
    print("BaggingClassifier prediction:      " + str(y_pred_bag))
    y_pred_ada = adaboostClf.predict(x_test)
    print("AdaBoostClassifier prediction:     " + str(y_pred_ada))
    y_pred_rf = rfClf.predict(x_test)
    print("RandomForestClassifier prediction: " + str(y_pred_rf))
    y_pred_v = vClf.predict(x_test)
    print("VotingClassifier prediction:       " + str(y_pred_v))

    print("BaggingClassifier cross val: " + str(test_cross_validation(bagClf, x_val, y_val)))
    print("AdaBoostClassifier cross val: " + str(test_cross_validation(adaboostClf, x_val, y_val)))
    print("RandomForestClassifier cross val: " + str(test_cross_validation(rfClf, x_val, y_val)))
    print("VotingClassifier cross val: " + str(test_cross_validation(vClf, x_val, y_val)))

    output_my_results(y_pred_v)

if __name__ == '__main__':
    main()


# [1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1]
