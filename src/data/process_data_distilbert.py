"""
Данный файл обрабатывает данные для постов пользователя с помощью нейросети.
Идея - закодировать посты в виде эмбедингов и использовать на них метод понижения пространства что бы использовать как фичи
для обучения ML модели. Использую лингвистическю модель DistilBertModel, т.к. она легче Bert и Roberta,
но не сильно уступает им в качестве. В качестве метода понижения пространства - PCA
"""

import pandas as pd
import torch
from sklearn.decomposition import PCA
from src.models.predict_nn import  get_model, PostDataset, get_embeddings_labels
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from src.features.build_features import transform_post


posts_info = pd.read_csv('../../data/raw/post.csv')

tokenizer, model = get_model('distilbert')

# создадим экземпляр класса для датасета
dataset = PostDataset(posts_info['text'].values.tolist(), tokenizer)
# укажем способ сбора батчей с паддингом
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#сформируем даталоадер для модели
loader = DataLoader(dataset, batch_size=32, collate_fn=data_collator, pin_memory=True, shuffle=False)

# проверим где будем проводить вычисления
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# создаем эмбединги постов
embeddings = get_embeddings_labels(model, loader).numpy()
# центрируем данные
centered = embeddings - embeddings.mean()
#уменьшаем размерность
pca = PCA(n_components=2,random_state = 74)
pca_decomp = pca.fit_transform(centered)

new_post_df = transform_post(posts_info,n_pca = 0, drop = True, ohe = True,lemma =False)
new_post_df = pd.merge(new_post_df, pd.DataFrame(pca_decomp,columns = ['PCA_1', 'PCA_2']),left_index=True, right_index=True,how='left')
new_post_df.to_csv('../../data/interim/post_transform_nn.csv', index=False)

