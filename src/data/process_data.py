from src.features.build_features import transform_user_with_drop_fich, transform_post
import pandas as pd

post = pd.read_csv('../../data/raw/post.csv')
user = pd.read_csv('../../data/raw/user.csv')

drop_col = []
ohe_col = ['os', 'source']  # Колонки для one hot
ohe_topic = True  # КОДИРУЕМ ЛИ TOPIC
count_city = True  # Кодируем ли счетчиком city
count_country = True  # Кодируем ли country
age_group = 'none'  # Кодируем ли возраст в группы
norm = True  # Проводим ли нормализацию
pca_num = 2  # Кол-во PCA в сете

user_transformd_ = transform_user_with_drop_fich(user,
                                                 drop_col=drop_col,
                                                 ohe_col=ohe_col,
                                                 count_city=count_city,
                                                 count_country=count_country,
                                                 age_group=age_group,
                                                 norm=norm)

post_transformd_ = transform_post(post,
                                  pca_num,
                                  drop=False,
                                  ohe=True,
                                  lemma=False)

user_transformd_.to_csv('../../data/interim/user_transform.csv', index=False)
post_transformd_.to_csv('../../data/interim/post_transform.csv', index=False)
