## ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ МОДЕЛИ И ПРЕДСКАЗАНИЯ ВЕРОЯТНОСТИ ##

import numpy as np
import pandas as pd
from Metrics import model_metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Протестировать модели с базовыми параметрами, вывести метрики для каждой
def test_base_models(X_train, y_train, X_test, y_test, RANDOM_SEED):
    models = [MLPClassifier, KNeighborsClassifier, XGBClassifier]
    for i in models:
        model = i(random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'{i}:')
        model_metrics(y_test,y_pred)
        print('-'*25)

# Подбор гиперпараметров и сохранение модели с лучшим показателем
def tune_model(model, params, X_train, y_train, RANDOM_SEED, name):
    # разбиение выборки для проверки
    folds = 3
    param_comb = 7
    # перекрестная валидация
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = RANDOM_SEED)
    # рандомизированный поиск по гиперпараметрам
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, 
                                        n_jobs=-1, cv=skf.split(X_train,y_train), verbose=0, random_state=RANDOM_SEED )
    # обучаю модель по лучшим подобранным гиперпараметрам
    random_search.fit(X_train, y_train)
    # сохраняю модель
    joblib.dump(random_search.best_estimator_, f'save_models/{name}_model.pkl')

# Формирование файла с вероятностью дефолта
def sub_create(path_model, X_test, client_id_file, model_name):
    model = joblib.load(path_model)
    y_pred = model.predict_proba(X_test)
    results_df = pd.DataFrame(data={'client_id':client_id_file['client_id'], 'default':y_pred[:,1]})
    results_df.to_csv(f'sub/{model_name}_sub.csv', index=False)

