from fastapi import FastAPI
from typing import List
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pydantic import BaseModel
import hashlib
import os
from catboost import CatBoostClassifier

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]



SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_model_path_test(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально.
        MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def get_model_path_control(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально.
        MODEL_PATH = '/workdir/user_input/model_control'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models_test():
    model_path = get_model_path_test("model_Cat_1500it_(2ml)_with_data")

    model = CatBoostClassifier()
    model.load_model(model_path, format='cbm')

    return model


def load_models_control():
    model_path = get_model_path_control("model_Cat_2000it_(2ml)_with_data_nn")

    model = CatBoostClassifier()
    model.load_model(model_path, format='cbm')

    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features_user() -> pd.DataFrame:
    user = batch_load_sql('SELECT * FROM "shestov_user_lesson_22_v2"')
    return user


def load_features_post_test() -> pd.DataFrame:
    post_sort = batch_load_sql("""SELECT * from "shestov_post_lesson_22_v1.1"
     """)
    return post_sort


def load_features_post_control() -> pd.DataFrame:
    post_sort = batch_load_sql("""SELECT * from "shestov_post_lesson_22_2_pca_nnn"
     """)
    return post_sort


model_cat_test = load_models_test()
model_cat_control = load_models_control()


user = load_features_user().drop(['index'], axis=1)

post_test = load_features_post_test().drop(['index'],axis=1)

post_control = load_features_post_control().drop(['index'],axis=1)

app = FastAPI()  #

def get_group(user, group_count = 2):
    value_str = str(user) + 'first_exp'
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    if  (value_num % group_count) == 0:
        return 'test'
    elif (value_num % group_count) == 1:
        return 'control'


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[Response]:
    result_list = []
    exp_group = get_group(id)
    if exp_group == 'test':
        post = post_test
        model_cat = model_cat_test

    elif exp_group == 'control':
        post = post_control
        model_cat = model_cat_control

    df_test = pd.merge(
        user[user['user_id'] == id],
        post, how='cross').drop(['user_id'], axis=1).set_index(['post_id'])
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

    return {'exp_group': exp_group,
            'recommendations': result_list}



