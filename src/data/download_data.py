"""
Файл достает из базы данных данные из таблиц user, post и feed
Данные в таблице feed сильно смещены: соотношение рекций ползователя лайк/не_лайк ~ 1/10.
Поэтому искуственно сбалансируем данные вытащив по 1млн каждой реакции.

"""
import pandas as pd
from tqdm import tqdm
import url
conn_uri = url.conn_uri

def get_user(conn_uri_):
    return pd.read_sql("""SELECT * FROM public.user_data """ , conn_uri_ )

def get_post(conn_uri_):
    return pd.read_sql("""SELECT * FROM public.post_text_df""", conn_uri_ )
     
def get_feed(conn_uri_, n_feed): 
    return pd.read_sql(f"""SELECT * FROM public.feed_data limit {n_feed}""", conn_uri_ )   

def get_feed_2(conn_uri_, n_feed): 
    return pd.read_sql(f"""SELECT * FROM public.feed_data  where  action = 'view' limit {n_feed}""", conn_uri_ )   

def get_feed_like(conn_uri_, n_feed): 
    return pd.read_sql(f"""SELECT * FROM public.feed_data  where  (action = 'view' and target= 1) limit {n_feed}""", conn_uri_ )   

def get_feed_not_like(conn_uri_, n_feed): 
    return pd.read_sql(f"""SELECT * FROM public.feed_data  where  (action = 'view'and target= 0) limit {n_feed}""", conn_uri_ )   

# 1 млн лайков
feed_like = get_feed_like(conn_uri, 1000000)
# 1 млн пустой реакции
feed_not_like = get_feed_not_like(conn_uri, 1000000)
#датасет 50/50 лайки/просмотры
feed50_50 = pd.concat((feed_like, feed_not_like), axis =0).reset_index() 
feed50_50.to_csv('../../data/raw/feed50_50_2mil.csv',index=False)

post_save = get_post(conn_uri) #чистый пост
user_save = get_user(conn_uri) #чистый юзер
post_save.to_csv('../../data/raw/post.csv', index=False)
user_save.to_csv('../../data/raw/user.csv', index=False)