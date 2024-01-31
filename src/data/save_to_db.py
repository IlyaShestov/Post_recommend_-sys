import pandas as pd
import url
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

conn_uri = url.conn_uri

post = pd.read_csv('../../data/interim/post_transform.csv.csv')
post_nn = pd.read_csv('../../data/interim/post_transform_nn.csv.csv')
user = pd.read_csv('../../data/interim/user_transform.csv.csv')

engine = create_engine(conn_uri)
post.to_sql('post_transform', con=engine)  # записываем таблицу
post_nn.to_sql('post_transform_nn', con=engine)  # записываем таблицу
user.to_sql('user_transform', con=engine)  # записываем таблицу
