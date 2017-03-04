import pandas as pd


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['bathrooms', 'bedrooms', 'month', 'day']
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


def _get_month_day_from_posting_date(design_matrix):
    """Capture month & day from date of posting"""
    design_matrix['day'] = design_matrix['created'].dt.day
    design_matrix['month'] = design_matrix['created'].dt.month
    design_matrix.drop('created', axis=1, inplace=True)
    return design_matrix


def clean_design_matrix(design_matrix, train=False):
    """Clean/transform design matrix"""
    if train:
        design_matrix = design_matrix[['interest_level', 'price', 'bathrooms',
                                       'bedrooms', 'created']]
    else:
        design_matrix = design_matrix[['price', 'bathrooms', 'bedrooms',
                                       'created']]
    design_matrix = _get_month_day_from_posting_date(design_matrix)
    design_matrix = _convert_data_types(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix,
                                                             train)
    return design_matrix
