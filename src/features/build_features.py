import pandas as pd
import numpy as np
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.decomposition import PCA
import nltk

np.random.seed(74)
nltk.download('wordnet')


def transform_post(post, n_pca=20, drop=True, ohe=True, lemma=True):
    """
    Функция получает на вход таблицу пост post.
    Опчионально проводит OneHotEncoding типов постов,
    Лемматизацию текстов, Tf-Idf и накладывает на полученые вектора  PCA

    :n_pca: кол-во изерений в новом пространстве, если 0 не проводит pca b tf-idf
    :drop: = True/False дропать ли колонку текст
    :ohe: = True/False проводить ли  OneHotEncoding колонки topic
    :lemma: =True/False проводить ли лемматизацию


    """
    post_ = post.copy()
    wnl = WordNetLemmatizer()

    # лемматизация текста
    if lemma:
        def preprocessing(line, token=wnl):
            line = line.lower()
            line = re.sub(r"[{}]".format(string.punctuation), " ", line)
            line = line.replace('\n\n', ' ').replace('\n', ' ')
            line = ' '.join([token.lemmatize(x) for x in line.split(' ')])
            return line
    else:
        preprocessing = None

    # учу OHE_Post
    if ohe:
        one_hot = pd.get_dummies(post_['topic'], prefix='topic', drop_first=True)
        post_ = pd.concat((post_.drop(['topic'], axis=1), one_hot), axis=1)
        post_transformd = post_
    elif not ohe:
        post_transformd = post_.drop(['topic'], axis=1)

    # tf-idf+ pca
    if n_pca > 0:
        # провожу tf-idf для Post
        tf = TfidfVectorizer(stop_words='english',
                             preprocessor=preprocessing,
                             min_df=5)  # создаю экземпляр класса
        tf_idf_ = tf.fit_transform(post_['text'])  # учу класс
        tf_idf_ = tf_idf_.toarray() - tf_idf_.mean()  # центрируем данные

        list_col_pca = [f"PCA_{nn}" for nn in range(1, n_pca + 1)]
        pca = PCA(n_components=n_pca, random_state=74)

        # создаю экземплря PCA
        PCA_dataset = pca.fit_transform(tf_idf_)  # провожу PCA
        PCA_dataset = pd.DataFrame(PCA_dataset, columns=list_col_pca, index=post.index)

        # Трансформирую Post
        post_transformd = pd.concat((post_transformd, PCA_dataset), axis=1)
        if drop:
            post_transformd = post_transformd.drop(['text'], axis=1)

    else:
        if drop:
            post_transformd = post_transformd.drop(['text'], axis=1)

    return post_transformd


def transform_feed(feed):  # предобрабатывает таблицу feed
    feed_ = feed.copy()
    feed_ = feed_.drop_duplicates()
    return feed_


def transform_user_with_drop_fich(user,
                                  drop_col=None,
                                  ohe_col=None,
                                  count_city=True,
                                  count_country=True,
                                  age_group='group',
                                  norm=True):
    """
    Функция получает на вход таблицу пост user.
    Опционально проводит дропает колонки, проводит OnehotEncoding и CounterEncoding
    а так же может раскодировать возвраст по группам
    :drop_col:колонки для дропа
    :ohe_col: колонки для OnehotEncoding
    :count_city: и :count_country: (Bool) проводить ли CounterEncoding по колонкам city и country
    :age_group: 'group'/None кодировать ли возвраст
    :norm: (Bool) нормализовать ли данные при CounterEncoding
    """
    if ohe_col is None:
        ohe_col = []
    if drop_col is None:
        drop_col = []
    user_ = user.copy().drop(drop_col, axis=1)  # убираю колонки дропа
    # Новый столбец с категориями возраста
    bins = [0, 18, 30, 60, 120]

    if 'age' not in drop_col and (age_group == 'group'):
        user_['age_group'] = pd.cut(user_['age'], bins, labels=['0-17', '18-30', '30-60', '60+'])
        user_ = user_.drop(['age'], axis=1)
    elif 'age' not in drop_col and (age_group == 'count'):
        counter_age = CountEncoder(cols=['age'],
                                   return_df=True,
                                   normalize=True)
        user_ = counter_age.fit_transform(user_)
    elif 'age' not in drop_col and (age_group == 'none'):
        user_ = user_

    # учу CounterEncoder_User

    if ('city' not in drop_col) and (count_city == True):
        counter_user = CountEncoder(cols=['city'],
                                    return_df=True,
                                    normalize=norm)
        user_ = counter_user.fit_transform(user_)

    if ('country' not in drop_col) and (count_country == True):
        counter_country = CountEncoder(cols=['country'],
                                       return_df=True,
                                       normalize=norm)
        user_ = counter_country.fit_transform(user_)

    # OHE_User
    for col in ohe_col:
        one_hot = pd.get_dummies(user_[col], prefix=col, drop_first=True)
        user_ = pd.concat((user_.drop(col, axis=1), one_hot), axis=1)

    return user_


def get_data_features(df):
    """
    Принимает датафрейм
    Выделяет из колонки timestamp час, месяц и день недели
    Возвращет датафрейм временными фичами
    """
    df_with_data = df.copy()
    df_with_data['hour'] = pd.to_datetime(df_with_data['timestamp']).apply(lambda x: x.hour)
    df_with_data['month'] = pd.to_datetime(df_with_data['timestamp']).apply(lambda x: x.month)
    df_with_data['dayofweek'] = pd.to_datetime(df_with_data['timestamp']).apply(lambda x: x.dayofweek)
    return df_with_data
