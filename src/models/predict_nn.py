from transformers import DistilBertModel  # https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_model(model_name):
    """
    Функция возвращает токенайзер и предобученую модель по ключу 'distilbert'
    """
    assert model_name in ['distilbert']

    checkpoint_names = {'distilbert': 'distilbert-base-cased' } # https://huggingface.co/distilbert-base-cased}

    model_classes = {'distilbert': DistilBertModel}

    return AutoTokenizer.from_pretrained(checkpoint_names[model_name]), model_classes[model_name].from_pretrained(checkpoint_names[model_name])

# Напишем класс датасет для постов
class PostDataset(Dataset):
    """
    Класс обрабатывает полученый текст с помощью токнайзера 'distilbert'
    для использования в этой моделе
    """
    def __init__(self, texts, tokenizer):
        super().__init__()

        self.texts = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return {'input_ids': self.texts['input_ids'][idx], 'attention_mask': self.texts['attention_mask'][idx]}

    def __len__(self):
        return len(self.texts['input_ids'])

@torch.inference_mode()
def get_embeddings_labels(model, loader):
    """
    Функция получает на вход нейросеть и лоадер с пердобработаными постами
    Проводите обработку лоадера в режиме применения
    На выходе получаем эмбединги для каждого поста
    """
    model.eval()

    total_embeddings = []

    for batch in tqdm(loader):
        batch = {key: batch[key].to(device) for key in ['attention_mask', 'input_ids']}

        embeddings = model(**batch)['last_hidden_state'][:, 0, :]

        total_embeddings.append(embeddings.cpu())

    return torch.cat(total_embeddings, dim=0)