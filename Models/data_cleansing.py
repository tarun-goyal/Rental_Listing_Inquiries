import pandas as pd


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['bathrooms', 'bedrooms', 'month']
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
    design_matrix['year'] = design_matrix['created'].dt.year
    design_matrix['weekday'] = design_matrix['created'].dt.weekday
    design_matrix.drop('created', axis=1, inplace=True)
    return design_matrix


def _length_of_variables(design_matrix):
    design_matrix["num_photos"] = design_matrix["photos"].apply(len)
    design_matrix["num_features"] = design_matrix["features"] \
        .apply(lambda x: len(x.split(" ")))
    design_matrix["num_description_words"] = design_matrix["description"] \
        .apply(lambda x: len(x.split(" ")))
    design_matrix.drop(['features', 'photos', 'description'],
                       axis=1, inplace=True)
    return design_matrix


def _convert_classes_into_floats(design_matrix):
    design_matrix['interest_level'] = design_matrix[
        'interest_level'].apply(
        lambda x: 0 if x == 'low' else 1 if x == 'medium' else 2)
    return design_matrix


def clean_design_matrix(design_matrix, train=False):
    """Clean/transform design matrix"""
    design_matrix.drop(['display_address'], axis=1, inplace=True)
    design_matrix = _get_time_features_from_posting_date(design_matrix)
    design_matrix = _length_of_variables(design_matrix)
    design_matrix = _convert_data_types(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix,
                                                             train)
    # if train:
    #     _convert_classes_into_floats(design_matrix)
    print design_matrix.shape
    return design_matrix
