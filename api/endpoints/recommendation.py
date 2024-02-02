from datetime import datetime
import pandas as pd

#def recommended_posts(id: int, time: datetime, limit: int = 5):
def recommended_posts(id: int, time, user_transformd_ ,post_sort, model_cat, limit: int = 5):
    """
    Функция принимает на вход id  пользователя, время реакции пользователья на пост и лимит рекомендаций
    Возвращает пользователю рекомендации в количестве limit
    """
    result_list = []
    df_test = pd.merge(
        user_transformd_[user_transformd_['user_id'] == id],
        post_sort, how='cross').drop(['user_id'], axis=1).set_index(['post_id'])
    df_test['hour'] = pd.to_datetime(time).hour
    df_test['month'] = pd.to_datetime(time).month
    df_test['dayofweek'] = pd.to_datetime(time).dayofweek
    predict_pr = model_cat.predict_proba(df_test.drop(['text', 'topic'], axis=1))
    result = pd.DataFrame(predict_pr, index=df_test.index).drop([0], axis=1).sort_values(by=1, ascending=False)[:limit]

    for i in range(limit):
        id_ = int(result.index[-1 + i])
        result_list.append(

            {"id": id_,
             "text": str(df_test.loc[id_]['text']),
             "topic": str(df_test.loc[id_]['topic'])}

        )

    return result_list