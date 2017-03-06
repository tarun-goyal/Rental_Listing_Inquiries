import pandas as pd


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['bathrooms', 'bedrooms']
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
    """Extract house features from the list provided"""
    feature_list = []
    design_matrix.reset_index(inplace=True)
    for row in range(design_matrix.shape[0]):
        feature_list.extend(design_matrix.loc[row, 'features'])
    feature_list = list(set(feature_list))
    return feature_list


def _street_or_avenue(design_matrix):
    """Extract places/areas from the displayed address"""
    design_matrix['Avenue'] = design_matrix['display_address']\
        .str.contains("Ave|ave|AVE")
    design_matrix['Street'] = design_matrix['display_address'] \
        .str.contains("St|st|ST")
    design_matrix['Place'] = design_matrix['display_address'] \
        .str.contains("Place|place|PLACE")
    design_matrix['Boulevard'] = design_matrix['display_address'] \
        .str.contains("Boulevard|Blvd|blvd|BOULEVARD|BLVD")
    design_matrix['Broadway'] = design_matrix['display_address'] \
        .str.contains("Broad|broad|BROAD")
    design_matrix['Parkway'] = design_matrix['display_address'] \
        .str.contains("Park|park|PARK")
    design_matrix['Road'] = design_matrix['display_address'] \
        .str.contains("Road|road|ROAD")
    design_matrix['Riverside'] = design_matrix['display_address'] \
        .str.contains("River|Water|RIVER|WATER|river|water")
    design_matrix['W'] = design_matrix['display_address'] \
        .str.contains("W")
    design_matrix['E'] = design_matrix['display_address'] \
        .str.contains("E")
    design_matrix.drop('display_address', axis=1, inplace=True)
    return design_matrix


def _info_from_other_features(design_matrix):
    design_matrix["num_photos"] = design_matrix["photos"].apply(len)
    design_matrix["num_features"] = design_matrix["features"].apply(len)
    design_matrix["num_description_words"] = design_matrix[
        "description"].apply(lambda x: len(x.split(" ")))
    design_matrix.drop(['photos', 'features', 'description'],
                       axis=1, inplace=True)
    return design_matrix


def clean_design_matrix(design_matrix, train=False):
    """Clean/transform design matrix"""
    if train:
        design_matrix = design_matrix[[
            'interest_level', 'price', 'bathrooms', 'bedrooms', 'created',
            'display_address', 'latitude', 'longitude', 'photos', 'features',
            'description']]
    else:
        design_matrix = design_matrix[[
            'price', 'bathrooms', 'bedrooms', 'created',
            'display_address', 'latitude', 'longitude', 'photos', 'features',
            'description']]
    design_matrix = _street_or_avenue(design_matrix)
    design_matrix = _get_time_features_from_posting_date(design_matrix)
    design_matrix = _info_from_other_features(design_matrix)
    design_matrix = _convert_data_types(design_matrix)
    # design_matrix = _extract_features(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix,
                                                             train)
    return design_matrix
