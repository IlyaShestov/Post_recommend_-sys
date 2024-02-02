from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from src.data.url import conn_uri

engine = create_engine(conn_uri)

pd.read_sql('SELECT * FROM my_favourite_table', con=engine)
def load_features_user() -> pd.DataFrame:
    user = pd.read_sql('SELECT * FROM "shestov_user_lesson_22_v2"', con=engine)
    return user


def load_features_post_new() -> pd.DataFrame:
    post_sort = pd.read_sql('SELECT * from "shestov_post_lesson_22_v1.1"', con=engine)
    return post_sort
