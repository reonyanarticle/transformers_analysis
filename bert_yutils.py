from transformers import BertJapaneseTokenizer, BertModel
import torch
import re
from torch.utils.dlpack import to_dlpack
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

available = False  # type: bool
try:
    import cupy as xp
    available = True
except Exception as e:
    import numpy as xp
    _resolution_error = e


def check_cupy_available():
    """cupyがつかえるかどうか確認するための関数です．
    """
    print('Usage of cupy:{0}'.format(available))


def cosine_similarity(vec1, vec2):
    xp_vec1 = xp.array(vec1)
    xp_vec2 = xp.array(vec2)
    return xp.dot(xp_vec1, xp_vec2) / (xp.linalg.norm(xp_vec1) *
                                       xp.linalg.norm(xp_vec1))


class BertAnalytics(object):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, model, tokenizer, window_size=128):
        self.model = model.to(self.DEVICE)
        self.tokenizer = tokenizer
        self.window_size = window_size

    def get_embedding(self, text, show_tokens=False):
        bert_tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + bert_tokens[:(self.window_size - 2)] + ['[SEP]'])
        tokens_tensor = torch.tensor(ids).reshape(1, -1).to(self.DEVICE)
        self.model.eval()

        with torch.no_grad():
            all_encoder_laylers, _ = self.model(tokens_tensor)
            embedding = all_encoder_laylers[0]

        if available:
            xp_embedding = xp.fromDlpack(to_dlpack(embedding))

        else:
            xp_embedding = embedding.numpy()

        if xp_embedding.shape[0] < self.window_size:
            xp_embedding = xp.concatenate([xp_embedding, xp.zeros(((self.window_size - xp_embedding.shape[0]), 768))],0)

        if show_tokens:
            return (xp_embedding, ['[CLS]'] + bert_tokens[:(self.window_size - 2)] + ['[SEP]'])
        else:
            return xp_embedding

    def weight_mean(self, third_array, tokenized_tokens_list):
        join_sentence_list = [' '.join(line) for line in tokenized_tokens_list]
        vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)')
        tfidf_vectors = vectorizer.fit_transform(join_sentence_list)
        feature_tokens = vectorizer.get_feature_names()
        weights = tfidf_vectors.toarray()

        new_tokenized_tokens_list = []
        for tokenized_sentence in tokenized_tokens_list:
            tmp_list = []
            for token in tokenized_sentence:
                tmp_list.append(re.sub('#','', token))
            new_tokenized_tokens_list.append(tmp_list)

        norm_weights_list = []
        for i, tokenized_sentence in enumerate(new_tokenized_tokens_list):
            norm_weights = []
            for token in tokenized_sentence:
                if token in feature_tokens:
                    norm_weights.append(weights[i][feature_tokens.index(token)])
                else:
                    norm_weights.append(0)

            if len(norm_weights) != self.window_size:
                norm_weights += (self.window_size - len(norm_weights))*[0]

            norm_weights_list.append(norm_weights)

        xp_weights = xp.array(norm_weights_list)
        x_list = [(((xp_weight * X.T).T).sum(axis=0) / (xp.array(xp_weight)).sum())
                  if (xp.array(xp_weight)).sum() != 0
                  else ((xp_weight * X.T).T).sum(axis=0)
                  for X, xp_weight in zip(third_array, xp_weights)]
        return xp.concatenate(x_list)