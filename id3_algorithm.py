import sys

import numpy as np
import pandas as pd


def calc_entropy(data, classificator_name, classificator_values):
    rows_count = data.shape[0]
    entropy = 0

    for c_value in classificator_values:
        c_count = data[data[classificator_name] == c_value].shape[0]
        c_entropy = 0
        if c_count != 0:
            c_probability = c_count / rows_count
            c_entropy = -c_probability * np.log2(c_probability)
        entropy += c_entropy

    return entropy


def calc_inf_gain(feature_name, data, classificator_name, classificator_values):
    feature_values = data[feature_name].unique()
    rows_count = data.shape[0]
    feature_inf_gain = 0

    for f_value in feature_values:
        f_value_data = data[data[feature_name] == f_value]
        f_value_count = f_value_data.shape[0]
        f_value_entropy = calc_entropy(f_value_data, classificator_name, classificator_values)
        f_value_probability = f_value_count / rows_count
        feature_inf_gain += f_value_probability * f_value_entropy

    return calc_entropy(data, classificator_name, classificator_values) - feature_inf_gain


def find_max_inf_feature(data, classificator_name, classificator_values):
    features = data.columns.drop(classificator_name)
    max_inf_gain = -1
    max_inf_feature = None

    for feature in features:
        feature_inf_gain = calc_inf_gain(feature, data, classificator_name, classificator_values)
        if max_inf_gain < feature_inf_gain:
            max_inf_gain = feature_inf_gain
            max_inf_feature = feature

    return max_inf_feature


def generate_subtree(feature_name, data, classificator_name, classificator_values):
    feature_dict = data[feature_name].value_counts(sort=False)
    tree = {}

    for f_value, f_value_count in feature_dict.iteritems():
        f_value_data = data[data[feature_name] == f_value]
        is_one_value_node = False

        for c_value in classificator_values:
            c_value_count = f_value_data[f_value_data[classificator_name] == c_value].shape[0]
            if c_value_count == f_value_count:
                tree[f_value] = c_value
                data = data[data[feature_name] != f_value]
                is_one_value_node = True

        if not is_one_value_node:
            tree[f_value] = '?'

    return tree, data


def generate_tree(root, previous_feature_value, data, classificator_name, classificator_values, max_depth, current_depth=0):
    if data.shape[0] != 0 and current_depth < max_depth:
        current_depth += 1
        max_inf_feature = find_max_inf_feature(data, classificator_name, classificator_values)
        tree, data = generate_subtree(max_inf_feature, data, classificator_name, classificator_values)

        if previous_feature_value is not None:
            root[previous_feature_value] = {}
            root[previous_feature_value][max_inf_feature] = tree
            next_feature = root[previous_feature_value][max_inf_feature]
        else:
            root[max_inf_feature] = tree
            next_feature = root[max_inf_feature]

        for feature_value, branch in list(next_feature.items()):
            if branch == '?':
                f_value_data = data[data[max_inf_feature] == feature_value]
                generate_tree(next_feature, feature_value, f_value_data, classificator_name, classificator_values, max_depth, current_depth)


def id3(train_data, classificator_name, max_depth):
    data = train_data.copy()
    tree = {}
    classificator_values = data[classificator_name].unique()
    generate_tree(tree, None, data, classificator_name, classificator_values, max_depth)
    return tree


