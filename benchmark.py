import numpy as np
import pandas as pd

from util import COUPLING_TYPES
from make_predictions import load_partitions, train_model, get_feature_importances, drop_low_frequency_cycle_columns


def main():
    benchmarks = {}
    for coupling_type in COUPLING_TYPES:
        benchmarks[coupling_type] = process_coupling_type(coupling_type)
    benchmarks = pd.Series(benchmarks).sort_values()
    print(benchmarks)
    print(benchmarks.mean())


def process_coupling_type(coupling_type):
    print('coupling_type', coupling_type)
    features = load_partitions('train', coupling_type)
    features = drop_low_frequency_cycle_columns(features)
    features = features.sample(frac=1)  # shuffle

    n_train = int(0.8 * len(features))
    train = features.iloc[:n_train:]
    test = features.iloc[n_train::]

    y_train = train['scalar_coupling_constant']
    X_train = train.drop(['scalar_coupling_constant'], axis=1)

    model = train_model(X_train, y_train)
    print(
        get_feature_importances(model, X_train)
            .head(20)
            .round(4)
    )

    y_test = test['scalar_coupling_constant']
    X_test = test.drop(['scalar_coupling_constant'], axis=1)
    y_pred = model.predict(X_test)

    mae = np.mean(np.abs(y_test - y_pred))
    print('mae', mae)

    log_mae = np.log(mae)
    print('log_mae', log_mae)

    return log_mae


__name__ == '__main__' and main()
