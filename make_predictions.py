import pickle
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
    return pd.concat(
        [pd.read_pickle(path)
         for path in
         tqdm(glob(f'data/partitions/features/{name}/{coupling_type}/*.p'))],
        axis=0).fillna(0.0).set_index('id')


def process_coupling_type(coupling_type):
    train = load_partitions('train', coupling_type)
    y_train = train['scalar_coupling_constant']
    X_train = train.drop(['scalar_coupling_constant'], axis=1)

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
    print(
        pd.Series(dict(zip(X_train.columns,
                           model.feature_importances_)))
            .sort_values(ascending=False)
            .head(20)
            .round(4)
    )

    X_test = load_partitions('test', coupling_type)
    for column in (set(X_train.columns) - set(X_test.columns)):
        X_test[column] = 0
    X_test = X_test[X_train.columns]

    y_pred = model.predict(X_test)

    for i, y in zip(X_test.index, y_pred):
        yield {'id': i, 'scalar_coupling_constant': round(y, 5)}

__name__ == '__main__' and main()