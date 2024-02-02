from src.data.download_processed_data import load_features_user, load_features_post_new
from src.models.download_model import load_models
import uvicorn
from fastapi import FastAPI
from api.endpoints.recommendation import recommended_posts
from api.schemas import PostGet
from typing import List

print('start')
#загружаем данные для работы модели
user_transformd_ = load_features_user().drop(['index'],axis=1)

print('download user')

post_sort = load_features_post_new().drop(['index'],axis=1)
print('download post')

#загружаем модель
model_path = ("models/model_Cat_2000it_(2ml)_tfidf")
model_cat = load_models(model_path)

print('download model')


app = FastAPI()  # экземпляр fastApi
# создаем эндпоинт
@app.get("/post/recommendations", response_model=List[PostGet])
async def get_recommendations(id: int, time, limit: int = 5):
    return recommended_posts(id, time, user_transformd_, post_sort, model_cat, limit,)

if __name__ == "__main__":
    uvicorn.run("run:app", port=5000, log_level="info")
    print('start server')