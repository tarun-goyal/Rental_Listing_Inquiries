import pandas as pd
import numpy as np


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['bathrooms', 'bedrooms', 'month', 'day', 'hour']
    for column in conversion_list:
        design_matrix[column] = design_matrix[column].apply(str)
    return design_matrix


def _create_dummies_for_categorical_features(design_matrix, train=False):
    """Create dummies for categorical features"""
    feature_types = dict(design_matrix.dtypes)
    categorical_features = [feature for feature, type in feature_types
                            .iteritems() if type == 'object']
    if train:
        categorical_features.remove('interest_level')
    design_matrix = pd.get_dummies(design_matrix, prefix=categorical_features,
                                   columns=categorical_features)
    return design_matrix


def _get_time_features_from_posting_date(design_matrix):
    """Capture month & day from date of posting"""
    design_matrix['day'] = design_matrix['created'].dt.day
    design_matrix['month'] = design_matrix['created'].dt.month
    design_matrix['hour'] = design_matrix['created'].dt.hour
    design_matrix.drop('created', axis=1, inplace=True)
    return design_matrix


def _extract_features(design_matrix):
    feature_list = []
    design_matrix.reset_index(inplace=True)
    for row in range(design_matrix.shape[0]):
        feature_list.extend(design_matrix.loc[row, 'features'])
    feature_list = list(set(feature_list))
    return feature_list


def _street_or_avenue(design_matrix):
    design_matrix['Avenue'] = design_matrix['display_address']\
        .str.contains("Ave")
    design_matrix['Street'] = design_matrix['display_address'] \
        .str.contains("St")
    design_matrix.drop('display_address', axis=1, inplace=True)
    return design_matrix


def clean_design_matrix(design_matrix, train=False):
    """Clean/transform design matrix"""
    if train:
        design_matrix = design_matrix[[
            'interest_level', 'price', 'bathrooms', 'bedrooms', 'created',
            'display_address']]
    else:
        design_matrix = design_matrix[[
            'price', 'bathrooms', 'bedrooms', 'created',
            'display_address']]
    design_matrix = _street_or_avenue(design_matrix)
    design_matrix = _get_time_features_from_posting_date(design_matrix)
    design_matrix = _convert_data_types(design_matrix)
    # design_matrix = _extract_features(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix,
                                                             train)
    return design_matrix
