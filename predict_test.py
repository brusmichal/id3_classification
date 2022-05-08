import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from id3_algorithm import id3


def find_prediction(tree, row):
    if not isinstance(tree, dict):
        return tree
    else:
        current_feature = next(iter(tree))
        feature_value = row[current_feature]
        if feature_value in tree[current_feature]:
            return find_prediction(tree[current_feature][feature_value], row)
        else:
            return '?'


def predict(tree, data):
    predictions = []
    for i, row in data.iterrows():
        predictions.append(find_prediction(tree, row))
    return predictions


def test():
    train_data = pd.read_csv('breast-cancer.data')
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xy_train = X_train.assign(irradiat=y_train)
    xy_test = X_test.assign(irradiat=y_test)

    tree = id3(xy_train, 'irradiat', 5)
    predictions = predict(tree, xy_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print(f"Actual values: {list(y_test)}")
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")


