## ФУНКЦИЯ ГЕНЕРАЦИИ ПРИЗНАКОВ ##

import numpy as np
import pandas as pd

# сгенерировать признаки для датафрейма, возвращает полный датафрейм с новыми признаками
def feature_eng(df):
    # cколько дней прошло с 1-го запроса
    df['app_date'] = pd.to_datetime(df.app_date)
    df['app_date'] = df['app_date'].apply(lambda x: (x - df['app_date'].min()).days)

    # Дом далеко от работы
    df['home_work'] = 0
    df.loc[df['home_address'] == df['work_address'], 'home_work'] = 1

    # Количество отказных заявок/Количество запросов
    df['bki/decline'] = 0
    df.loc[(df['bki_request_cnt'] != 0) & (df['decline_app_cnt'] != 0), 'bki/decline'] = df['decline_app_cnt']/df['bki_request_cnt']

    # Категории возрастов
    df['age_category'] = ''
    df.loc[df['age'] <= 24, 'age_category'] = '18-24'
    df.loc[(df['age'] >= 25) & (df['age'] <= 34), 'age_category'] = '25-34'
    df.loc[(df['age'] >= 35) & (df['age'] <= 44), 'age_category'] = '35-44'
    df.loc[(df['age'] >= 45) & (df['age'] <= 54), 'age_category'] = '45-54'
    df.loc[(df['age'] >= 55) & (df['age'] <= 64), 'age_category'] = '55-64'
    df.loc[(df['age'] >= 65), 'age_category'] = '65+'

    # Доход больше среднего по категории возраста
    mean_income = df.groupby('age_category')['income'].mean().to_dict()
    df['mean_income_age_cat'] = df['age_category'].map(mean_income)
    df['inc_large_mean'] = 0
    df.loc[(df['income'] > df['mean_income_age_cat']), 'inc_large_mean'] = 1

    # Нормализованный доход по категории возраста
    df["normalized_income_minus_mean"] = (df.income - df.mean_income_age_cat)
    df["normalized_income_minus_mean"] = df["normalized_income_minus_mean"] + abs(min(df["normalized_income_minus_mean"]))

    # Больше среднего обращений в БКИ по категории возраста
    mean_bki = df.groupby('age_category')['bki_request_cnt'].mean().to_dict()
    df['mean_requests_age'] = df['age_category'].map(mean_bki)
    df['request_large_mean'] = 0
    df.loc[(df['bki_request_cnt'] > df['mean_requests_age']), 'request_large_mean'] = 1

    # Доход больше среднего по региону
    mean_income_rat = df.groupby('region_rating')['income'].mean().to_dict()
    df['mean_income_region'] = df['region_rating'].map(mean_income_rat)
    df['inc_large_mean_region'] = 0
    df.loc[(df['income'] > df['mean_income_region']), 'inc_large_mean_region'] = 1

    # Без отказов от банков, но имеет обращения в БКИ
    df['active_no_decline'] = 0
    df.loc[(df['decline_app_cnt'] == 0) & (df['bki_request_cnt'] != 0), 'active_no_decline'] = 1

    # Не имеет запросов в БКИ и не имеет отказов
    df['no_decline_request'] = 0
    df.loc[(df['decline_app_cnt'] == 0) & (df['bki_request_cnt'] == 0), 'no_decline_request'] = 1

    # Имеет доход выше среднего, но не имеет пометку "хорошая" работа (бизнес?)
    df['no_good_job_and_good_income'] = 0
    df.loc[(df['good_work'] == 0) & (df['income'] > np.mean(df['income'])), 'no_good_job_and_good_income'] = 1

    # Доход/Скоринг оценку
    df['income_per_score_bki'] = df['income'] / abs(df['score_bki'])

    return(df)
    












    
