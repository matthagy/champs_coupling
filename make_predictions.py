import pickle

import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from tqdm import tqdm

print('loading train')
train = pd.read_csv('data/train.csv', usecols=['id', 'type', 'scalar_coupling_constant'])
#print('loading test')
#test = pd.read_csv('data/test.csv')


def load_features(name, coupling_type):
    print(f'loading features {name} {coupling_type}')
    acc = []
    with open(f'data/features_{name}_{coupling_type}.p', 'rb') as fp:
        with tqdm() as t:
            while True:
                t.update()
                try:
                    obj = pickle.load(fp)
                except (EOFError, OSError):
                    break
                else:
                    acc.append(obj)
    return pd.DataFrame(acc).fillna(0.0).set_index('id')


predictions = []
for coupling_type in sorted(set(train['type'])):
    print('type', coupling_type)

    X_train = load_features('train', coupling_type)

    y_train = train.set_index('id')['scalar_coupling_constant'].ix[X_train.index]

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

    X_test = load_features('test', coupling_type)
    for column in (set(X_train.columns) - set(X_test.columns)):
        X_test[column] = 0
    X_test = X_test[X_train.columns]
    y_pred = model.predict(X_test)

    for i,y in zip(X_test.index, y_pred):
        predictions.append({'id': i, 'scalar_coupling_constant': round(y, 5)})

print('writing predictions')
predictions = pd.DataFrame(predictions)[['id', 'scalar_coupling_constant']]
predictions.to_csv('data/submission_features2.csv', index=False)
