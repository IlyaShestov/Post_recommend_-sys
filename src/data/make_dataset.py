import pandas as pd
from src.features.build_features import get_data_features, transform_feed


def concat_df(feed_transformd, user_transformd, post_transformd):  # feed user post
    """
    Мерджит три таблицы feed, user, post в одну
    """
    df = pd.merge(feed_transformd.sort_values(by='timestamp'),
                  user_transformd,
                  on='user_id',
                  how='left')
    df = pd.merge(df,
                  post_transformd,
                  on='post_id',
                  how='left')
    return df


def splitter(df):
    """
    Делит сет на трейн и тест 80/20 и выделяет лейблы
    Принимает датаферйм
    Возвращает
    :X_train,y_train данные для обучения и лейблы
    :X_test,y_test данные для валидации и лейблы
    """
    train = df.iloc[:-int(df.shape[0] * 0.2)].copy()
    test = df.iloc[-int(df.shape[0] * 0.2):].copy()

    train_id = train[['user_id', 'post_id']]
    test_id = test[['user_id', 'post_id']]

    X_train, X_test = train.drop(['target', 'timestamp', 'user_id', 'post_id', 'action'], axis=1), test.drop(
        ['target', 'timestamp', 'user_id', 'post_id', 'action'], axis=1)
    y_train, y_test = train['target'], test['target']

    return [X_train, y_train, X_test, y_test, train_id, test_id]


def get_train_test(user, post, feed):
    """ Функция принимает обработанные  таблицы  feed, user, post.
    Возвращает полностью подготовленные данные для обработки:
    :X_train,y_train данные для обучения и лейблы
    :X_test,y_test данные для валидации и лейблы
    """

    feed_transformd_ = transform_feed(feed)
    post_ = post.copy()
    post_ = post_.drop(['text'], axis=1)

    df_ = concat_df(feed_transformd_, user, post_)

    df_ = get_data_features(df_)  # добавляем фичи из времени

    return splitter(df_)
