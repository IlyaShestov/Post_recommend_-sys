import pandas as pd
from catboost import CatBoostClassifier
from src.data.make_dataset import get_train_test

post = pd.read_csv('../../data/interim/post_transform.csv')
user = pd.read_csv('../../data/interim/user_transform.csv')
feed = pd.read_csv('../../data/raw/feed50_50_2mil.csv', parse_dates=["timestamp"]).drop(['index'], axis=1)
post_nn = pd.read_csv('../../data/interim/post_transform_nn.csv')

train_test_list = get_train_test(user, post, feed)

X_train, y_train, X_test, y_test = train_test_list[0], train_test_list[1], train_test_list[2], train_test_list[3]

# учим модель
# cat boost на трансформированых категориях + PCA (drop  0 )
i = 2000

model_Cat = CatBoostClassifier(random_state=74)
model_Cat.set_params(iterations=i, random_state=74, loss_function='Logloss')

model_Cat.fit(
        X_train,
        y_train,

        verbose=1,
        plot=0
    )


model_Cat.save_model('../../models/model_Cat_2000it_(2ml)_tfidf', format="cbm")  # сохраняем модель

# учим вторую модель
train_test_list = get_train_test(user, post_nn, feed)

X_train, y_train, X_test, y_test = train_test_list[0], train_test_list[1], train_test_list[2], train_test_list[3]

model_Cat_nn = CatBoostClassifier(random_state=74)
model_Cat_nn.set_params(iterations=i, random_state=74, loss_function='Logloss')

model_Cat_nn.fit(
        X_train,
        y_train,

        verbose=1,
        plot=0
    )


model_Cat_nn.save_model('../../models/model_Cat_2000it_(2ml)_nn', format="cbm")  # сохраняем модель
