from catboost import CatBoostClassifier
def load_models(model_path):
    """
    Загружаем модель
    """
    model = CatBoostClassifier()  # parameters not required.
    model.load_model(model_path, format='cbm') # пример как можно загружать модели

    return model