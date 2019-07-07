from glob import glob

import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from tqdm import tqdm

from util import COUPLING_TYPES


def main():
    predictions = []
    for coupling_type in COUPLING_TYPES:
        predictions.extend(process_coupling_type(coupling_type))
    print('writing predictions')
    predictions = pd.DataFrame(predictions)[['id', 'scalar_coupling_constant']]
    predictions.to_csv('data/submission_features3.csv', index=False)


def load_partitions(name, coupling_type):
    print('loading', name, coupling_type)
    features = pd.concat(
        [pd.read_pickle(path)
         for path in
         tqdm(glob(f'data/partitions/features/{name}/{coupling_type}/*.p'))],
        axis=0).fillna(0.0).set_index('id')
    if '1J' in coupling_type:
        # drop redundant features for 1J since there is only one bond
        features = features.drop(['a0_bond_length', 'a1_bond_length',
                                  'bond_path_length', 'bond_vector_dot',
                                  'bond_vector_dot_norm'],
                                 axis=1)

    return features


def drop_low_frequency_cycle_columns(features):
    cycle_columns = [c for c in features.columns if '_cyc_' in c]
    cycle_counts = features[cycle_columns].sum(axis=0)
    low_frequency_cycle_columns = cycle_counts[cycle_counts < 50].index
    print('dropping', len(low_frequency_cycle_columns), 'low frequency cycle columns')
    features = features.drop(low_frequency_cycle_columns, axis=1)
    return features


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=50,
        criterion='mse',
        max_features='auto',
        max_depth=25,
        min_samples_split=1e-4,
        min_samples_leaf=1e-5,
        n_jobs=-1,
        verbose=10
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, X_train):
    return (pd.Series(dict(zip(X_train.columns,
                               model.feature_importances_)))
            .sort_values(ascending=False))


def process_coupling_type(coupling_type):
    train = load_partitions('train', coupling_type)
    train = drop_low_frequency_cycle_columns(train)

    y_train = train['scalar_coupling_constant']
    X_train = train.drop(['scalar_coupling_constant'], axis=1)

    model = train_model(X_train, y_train)

    print(get_feature_importances(model, X_train)
          .head(20)
          .round(4))

    X_test = load_partitions('test', coupling_type)
    for column in (set(X_train.columns) - set(X_test.columns)):
        X_test[column] = 0
    X_test = X_test[X_train.columns]

    y_pred = model.predict(X_test)

    for i, y in zip(X_test.index, y_pred):
        yield {'id': i, 'scalar_coupling_constant': round(y, 5)}


__name__ == '__main__' and main()
